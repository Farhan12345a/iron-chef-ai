from operator import add
from typing import Annotated, Literal
import os
import textwrap

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel
from typing_extensions import TypedDict

load_dotenv()

MAX_JUDGE_ROUNDS = 3

raw_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
model_name = raw_model_name.replace("models/", "", 1)

try:
    llm = init_chat_model(
        model=model_name,
        model_provider="google_genai",
        api_key=os.getenv("GEMINI_API_KEY"),
    )
except ImportError as e:
    raise ImportError(
        "Missing langchain-google-genai. Run: uv add langchain-google-genai"
    ) from e


# ---------- Structured outputs ----------

class JudgeVerdict(BaseModel):
    decision: Literal["classic_chef", "fusion_chef", "end"]
    tasting_note: str = ""
    winner: Literal["classic_chef", "fusion_chef", "tie", "undecided"] = "undecided"
    winner_rationale: str = ""


class FinalVerdict(BaseModel):
    """Forced final ruling — never undecided."""
    winner: Literal["classic_chef", "fusion_chef", "tie"]
    winner_rationale: str = ""


# ---------- State ----------

class CookOffState(TypedDict):
    dish: str                                              # the ingredient/theme
    classic_chef_dishes: Annotated[list[str], add]        # classic chef's proposals
    fusion_chef_dishes: Annotated[list[str], add]         # fusion chef's proposals
    tasting_note: str                                     # judge's follow-up challenge
    judge_rounds: int
    winner: Literal["classic_chef", "fusion_chef", "tie", "undecided"]
    winner_rationale: str


# ---------- Agents ----------

def classic_chef(state: CookOffState) -> Command[Literal["fusion_chef", "judge"]]:
    prompt = state["tasting_note"] or "Propose your dish for this cook-off."

    is_opening = (
        len(state["classic_chef_dishes"]) == 0 or len(state["fusion_chef_dishes"]) == 0
    ) and state["judge_rounds"] == 0

    system_msg = (
        f"You are a classically trained French chef competing in a cook-off. "
        f"The featured ingredient/theme is: {state['dish']}. "
        "Your style: precise technique, traditional recipes, elegant plating."
    )

    if not is_opening and state["fusion_chef_dishes"]:
        system_msg += (
            f"\nYour rival's latest dish: {state['fusion_chef_dishes'][-1]}\n"
            "Explain why your classical approach is superior."
        )

    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ])

    needs_fusion_first = (
        len(state["classic_chef_dishes"]) == 0 and len(state["fusion_chef_dishes"]) == 0
    ) or (len(state["fusion_chef_dishes"]) == 0 and state["judge_rounds"] == 0)

    next_node = "fusion_chef" if needs_fusion_first else "judge"

    return Command(
        goto=next_node,
        update={
            "classic_chef_dishes": [response.content],
            "tasting_note": "",
        },
    )


def fusion_chef(state: CookOffState) -> Command[Literal["classic_chef", "judge"]]:
    prompt = state["tasting_note"] or "Propose your dish for this cook-off."

    is_opening = (
        len(state["classic_chef_dishes"]) == 0 or len(state["fusion_chef_dishes"]) == 0
    ) and state["judge_rounds"] == 0

    system_msg = (
        f"You are a bold fusion chef competing in a cook-off. "
        f"The featured ingredient/theme is: {state['dish']}. "
        "Your style: unexpected flavor combos, global influences, creative presentation."
    )

    if not is_opening and state["classic_chef_dishes"]:
        system_msg += (
            f"\nYour rival's latest dish: {state['classic_chef_dishes'][-1]}\n"
            "Explain why your fusion approach is superior."
        )

    response = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ])

    needs_classic_first = (
        len(state["classic_chef_dishes"]) == 0 and len(state["fusion_chef_dishes"]) == 0
    ) or (len(state["classic_chef_dishes"]) == 0 and state["judge_rounds"] == 0)

    next_node = "classic_chef" if needs_classic_first else "judge"

    return Command(
        goto=next_node,
        update={
            "fusion_chef_dishes": [response.content],
            "tasting_note": "",
        },
    )


def judge(state: CookOffState) -> Command:
    rounds = state["judge_rounds"] + 1

    judge_llm = llm.with_structured_output(JudgeVerdict)
    verdict = judge_llm.invoke([
        {
            "role": "system",
            "content": (
                f"You are a Michelin-star judge evaluating a cook-off on theme: {state['dish']}. "
                f"Classic chef's dishes: {state['classic_chef_dishes']}. "
                f"Fusion chef's dishes: {state['fusion_chef_dishes']}. "
                "Decide: classic_chef, fusion_chef, or end. "
                "If end, set winner to classic_chef, fusion_chef, or tie and provide rationale. "
                "Otherwise set winner to undecided. "
                "Use tasting_note to challenge chefs further if needed."
            ),
        },
        {
            "role": "user",
            "content": "Return decision, tasting_note, winner, winner_rationale.",
        },
    ])

    # Don't let the judge end on round 1 — force at least one follow-up
    if rounds == 1 and verdict.decision == "end":
        next_chef = (
            "classic_chef"
            if len(state["classic_chef_dishes"]) <= len(state["fusion_chef_dishes"])
            else "fusion_chef"
        )
        tasting_note = (
            verdict.tasting_note.strip()
            or "Surprise us — incorporate an unexpected technique into your next dish."
        )
        current_winner = "undecided"
        rationale = "Need at least one more round before a final ruling."
    else:
        next_chef = verdict.decision
        tasting_note = verdict.tasting_note
        current_winner = verdict.winner
        rationale = verdict.winner_rationale

    # Hard cap on rounds
    if rounds >= MAX_JUDGE_ROUNDS:
        next_chef = "end"

    if next_chef != "end":
        current_winner = "undecided"
        rationale = ""

    next_node = END if next_chef == "end" else next_chef

    # Force a real winner if ending without one
    if next_chef == "end" and current_winner not in ("classic_chef", "fusion_chef", "tie"):
        forced_llm = llm.with_structured_output(FinalVerdict)
        forced = forced_llm.invoke([
            {
                "role": "system",
                "content": (
                    f"You are a Michelin judge giving a final ruling on theme: {state['dish']}. "
                    f"Classic chef: {state['classic_chef_dishes']}. "
                    f"Fusion chef: {state['fusion_chef_dishes']}. "
                    "Pick exactly one: classic_chef, fusion_chef, or tie. Never undecided."
                ),
            },
            {"role": "user", "content": "Return winner and winner_rationale."},
        ])
        current_winner = forced.winner
        rationale = forced.winner_rationale

    return Command(
        goto=next_node,
        update={
            "judge_rounds": rounds,
            "tasting_note": tasting_note,
            "winner": current_winner,
            "winner_rationale": rationale,
        },
    )


# ---------- Build graph ----------

workflow = StateGraph(CookOffState)
workflow.add_node("classic_chef", classic_chef)
workflow.add_node("fusion_chef", fusion_chef)
workflow.add_node("judge", judge)
workflow.add_edge(START, "classic_chef")

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ---------- Run ----------

def _pretty_block(title: str, items: list[str]) -> None:
    print(f"\n{title}")
    if not items:
        print("  (none)")
        return
    i = 1
    while i <= len(items):
        wrapped = textwrap.fill(items[i - 1], width=100, initial_indent="  ", subsequent_indent="  ")
        print(f"  [{i}] {wrapped.strip()}")
        i += 1


initial_state: CookOffState = {
    "dish": "black truffle",
    "classic_chef_dishes": [],
    "fusion_chef_dishes": [],
    "tasting_note": "",
    "judge_rounds": 0,
    "winner": "undecided",
    "winner_rationale": "",
}

config = {"configurable": {"thread_id": "chef-cookoff-1"}}
result = app.invoke(initial_state, config=config)

print("=== Cook-Off Final Results ===")
print(f"Theme ingredient: {result['dish']}")
print(f"Judge rounds:     {result.get('judge_rounds', 0)}")
print(f"Winner:           {result.get('winner', 'undecided')}")

if result.get("winner_rationale"):
    print(f"Rationale: {result['winner_rationale']}")
if result.get("tasting_note"):
    print(f"Last tasting note: {result['tasting_note']}")

_pretty_block("CLASSIC CHEF dishes:", result.get("classic_chef_dishes", []))
_pretty_block("FUSION CHEF dishes:", result.get("fusion_chef_dishes", []))

print("\n=== End of Cook-Off ===")
