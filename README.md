# Chef Cook-Off — LangGraph Multi-Agent System

A multi-agent debate system re-imagined as a competitive cook-off. Two AI chefs — a classically trained French chef and a bold fusion chef — compete over a featured ingredient. A Michelin-star judge evaluates each round, issues tasting-note challenges, and ultimately declares a winner.

Built on [LangGraph](https://github.com/langchain-ai/langgraph) with Google Gemini via `langchain-google-genai`.

---

## Cook-Off Flow

```
START
  |
  v
+--------------------+
|   Classic chef     |   opening: both speak once
+----------+---------+
           |
           v
+--------------------+
|   Fusion chef      |
+----------+---------+
           |
           v
+--------------------+
|      Judge         |
+----+----------+----+
     |          |          |
     v          v          v
 classic    fusion        END
 (follow-up) (follow-up)
     |          |
     +----+-----+
          v
        Judge
          ...
          v
         END
          |
          v
  Winner declared
  classic_chef / fusion_chef / tie
```

Three agents: **classic_chef**, **fusion_chef**, and a **judge**. The graph starts at `classic_chef`, then `fusion_chef` for the opening round, then `judge`. The judge can route the next turn to either chef or `END`.

LLMs can show **anchoring bias** when early information pulls later judgments. Here, the judge does not act until **both** sides have produced an opening, and the graph requires **at least one follow-up** before allowing an early stop — so the first real evaluation is never based on a single side alone.

---

## Project Structure

```
.
├── chef_cookoff.py      # Main script (convert from notebook or run directly)
├── .env                 # API keys (not committed)
└── README.md
```

---

## Setup

**1. Clone / navigate to the project folder**

```bash
cd three_agent_debate_langgraph   # or wherever your project lives
```

**2. Create a `.env` file**

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash     # optional, this is the default
```

**3. Install dependencies**

```bash
pip install langchain langgraph langchain-google-genai python-dotenv pydantic
```

Or if using `uv`:

```bash
uv add langchain langgraph langchain-google-genai python-dotenv pydantic
```

---

## Run

```bash
python3 chef_cookoff.py
```

**Expected output:**

```
=== Cook-Off Final Results ===
Theme ingredient: black truffle
Judge rounds:     2
Winner:           fusion_chef
Rationale: The fusion chef's unexpected black truffle miso butter...

CLASSIC CHEF dishes:
  [1] A classic French preparation...
  [2] In response to the fusion approach...

FUSION CHEF dishes:
  [1] A bold umami-forward creation...
  [2] Doubling down with a black truffle...

=== End of Cook-Off ===
```

---

## How It Works

| Component | Role |
|---|---|
| `classic_chef` | Argues for traditional technique; rebuts fusion chef each round |
| `fusion_chef` | Argues for creative fusion; rebuts classic chef each round |
| `judge` | Evaluates both sides, issues tasting-note challenges, declares winner |
| `MemorySaver` | Persists state across the graph execution via thread ID |
| `MAX_JUDGE_ROUNDS` | Hard cap (default: 3) prevents infinite loops |

### Anchoring bias protection

- The judge cannot end the debate on round 1 — a follow-up is forced
- If the judge reaches `MAX_JUDGE_ROUNDS` without a clear winner, a `FinalVerdict` call forces a definitive ruling (never `undecided`)

---

## Customization

**Change the ingredient/theme** — edit `initial_state` in `chef_cookoff.py`:

```python
initial_state: CookOffState = {
    "dish": "wagyu beef",   # ← change this
    ...
}
```

**Increase rounds** — change the cap at the top of the file:

```python
MAX_JUDGE_ROUNDS = 5
```

**Swap chef personas** — edit the `system_msg` strings inside `classic_chef()` and `fusion_chef()` to use any two competing styles (e.g. vegan vs carnivore, street food vs fine dining).

---

## Original Inspiration

Adapted from the `three_agent_debate` LangGraph notebook. Core graph structure, `Command`-based routing, anchoring-bias protection, and forced-verdict logic are all preserved — only the domain, agent names, and state schema have changed.
