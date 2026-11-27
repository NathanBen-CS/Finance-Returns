# Tier 0 network: a read-along guide to every line

This guide walks through the four files in this folder, line by line, as if you
were writing them yourself. It assumes you are new to PyTorch and to attention,
and that some of the denser Python lines look like magic. Nothing here is magic.

**How to use it:** open a file in one window and this guide in another. Every
section quotes the code (with its line numbers) and then explains it. Parts 1–4
are crash courses on the ideas the code uses. Read them first, or jump back to
them whenever a line confuses you. Part 9 follows one real event from your data
through the whole pipeline, with real numbers.

All numbers in this guide come from your actual data and were produced by running
the code (October 2026 version of the files).

---

## Contents

- [Part 0: The big picture](#part-0-the-big-picture)
- [Part 1: Python features used here](#part-1-python-features-used-here)
- [Part 2: NumPy and pandas, the bits used here](#part-2-numpy-and-pandas-the-bits-used-here)
- [Part 3: PyTorch from zero](#part-3-pytorch-from-zero)
- [Part 4: Attention from zero](#part-4-attention-from-zero)
- [Part 5: tier0_config.py, line by line](#part-5-tier0_configpy-line-by-line)
- [Part 6: tier0_prepare_data.py, line by line](#part-6-tier0_prepare_datapy-line-by-line)
- [Part 7: tier0_model.py, line by line](#part-7-tier0_modelpy-line-by-line)
- [Part 8: tier0_train.py, line by line](#part-8-tier0_trainpy-line-by-line)
- [Part 9: One real event's full journey](#part-9-one-real-events-full-journey)
- [Part 10: Running it and experimenting](#part-10-running-it-and-experimenting)
- [Part 11: Glossary](#part-11-glossary)

---

## Part 0: The big picture

### What are we trying to do?

Every row in `events.csv` is a company filing an 8-K with the SEC. For each one we
know how the stock reacted: `car_minus1_plus1` (we'll call it **CAR1**) is the
"cumulative abnormal return" over trading days -1, 0 and +1 around the filing, so
it measures how much the stock moved beyond the whole market, from the day before
to the day after.

The model's job: **given everything we knew about a company up to a filing, and
what the filing itself is, predict that filing's CAR1.** As a side job it also
predicts the `reaction_category` (Strong Negative … Strong Positive), which is
just CAR1 sorted into 5 buckets.

### What does the model get to see?

For each filing (call it the **current** filing), the model gets two things:

1. **The current filing's description**: what kind of filing it is, which 8-K
   items it lists, how late it was filed, and the company's profile (sector,
   size, etc.). 63 numbers in total.
2. **The company's history**: up to 128 earlier filings by the same company. For
   each one: what it was, how long ago it was, and *how the market reacted to
   it*, but only if that reaction was fully known before the current filing's
   own reaction window opened. 54 numbers per earlier filing.

It never sees the current filing's own reaction. That's the answer.

### The two steps (two scripts)

```
companies.csv ─┐
events.csv ────┼──► tier0_prepare_data.py ──► tier0_events.pkl ──► tier0_train.py ──► results
market_        │    (STEP 1: one row per       (141 MB of          (STEP 2: builds each
reactions.csv ─┘     event, numbers only)        number tables)      history, trains, tests)

            tier0_config.py: every setting, read by both scripts
            tier0_model.py:  the network itself, used by tier0_train.py
```

### The four files

| File | Lines | Job |
|---|---|---|
| `tier0_config.py` | 100 | All settings in one place: paths, the leakage rule, training knobs |
| `tier0_prepare_data.py` | 130 | Reads the 3 CSVs, turns every event into rows of numbers, saves them |
| `tier0_model.py` | 69 | The neural network: a small Transformer encoder plus two output heads |
| `tier0_train.py` | 234 | Builds each event's history, feeds batches to the model, trains, evaluates |

### The one idea everything is built around: no peeking at the answer

The biggest danger in this kind of model is **leakage**: accidentally showing the
model information it couldn't have known at prediction time, usually because
that information *contains* the answer. A leaky model looks brilliant in testing
and is worthless in reality.

The current filing's CAR1 uses its trading days -1, 0 and +1. So any number
computed from those days is off limits. An earlier filing's reaction can only be
shown if it was **fully measured before the current filing's day -1**. Part 5
works this out exactly; Part 9 shows it catching a real leak in your data.

---

## Part 1: Python features used here

Skim this once; come back when a line looks strange.

### Variables, lists, tuples, dicts

```python
x = 5                          # a variable
names = ["a", "b", "c"]        # a LIST: ordered, changeable. names[0] is "a"
point = (3, 4)                 # a TUPLE: like a list but can't be changed
prices = {"AAPL": 190, "MSFT": 410}   # a DICT: look things up by a key. prices["AAPL"] is 190
```

### Functions

```python
def add(a, b):        # "def" defines a function named add with two inputs
    return a + b      # "return" hands a value back to whoever called it

add(2, 3)             # -> 5
```

A function can also be defined *inside* another function (`log_or_zero` in the
prepare script does this). It can then use the outer function's variables.

### Docstrings and comments

```python
def f():
    """This is a docstring: a description of the function. Python ignores it when running."""
    x = 1   # this is a comment: everything after # is ignored by Python
```

A string in triple quotes at the very top of a file is the file's docstring.

### f-strings (formatted strings)

An `f` before a string lets you put Python expressions inside `{}`:

```python
n = 411685
f"we have {n} events"          # -> "we have 411685 events"
f"we have {n:,} events"        # -> "we have 411,685 events"   (:, adds thousands commas)
f"{0.123456:.4f}"              # -> "0.1235"                    (.4f = 4 decimal places)
f"{0.0123:+.4f}"               # -> "+0.0123"                   (+ always shows the sign)
f"{0.256:.1%}"                 # -> "25.6%"                     (% multiplies by 100, adds %)
f"{5:2d}"                      # -> " 5"                        (2d = whole number, 2 wide)
```

Two f-strings written next to each other in brackets are glued together:
`print(f"a {x} " f"b {y}")` prints one line.

### List comprehensions

A compact way to build a list with a loop:

```python
squares = [x * x for x in [1, 2, 3]]          # -> [1, 4, 9]
# exactly the same as:
squares = []
for x in [1, 2, 3]:
    squares.append(x * x)
```

Two `for`s in one comprehension run like nested loops, outer first:

```python
[f"{f}{p}" for p in (5, 21) for f in ("sin", "cos")]
# -> ['sin5', 'cos5', 'sin21', 'cos21']
```

### Conditional expressions ("inline if")

```python
x = "big" if n > 100 else "small"
```

### `dict(...)` and `**` (unpacking a dict into a function call)

```python
read = dict(keep_default_na=False, na_values=[""])
# is the same dict as {"keep_default_na": False, "na_values": [""]}

pd.read_csv("file.csv", **read)
# ** spreads the dict out into named arguments, so this is exactly:
pd.read_csv("file.csv", keep_default_na=False, na_values=[""])
```

Inside a dict literal, `**other_dict` copies all of `other_dict`'s entries in:

```python
a = {"x": 1}
b = {"y": 2, **a}     # -> {"y": 2, "x": 1}
```

### `*` (unpacking a list into a list or call)

```python
parts = [10, 20]
[1, *parts, 3]        # -> [1, 10, 20, 3]
```

### Tuple unpacking and `_`

```python
a, b = (1, 2)              # a = 1, b = 2
B, L, _ = (64, 63, 54)     # _ is a normal name, used by convention for "I don't need this one"
```

### `zip(*...)`: turning rows into columns

`zip` pairs up items from several lists. With `*`, it "transposes" a list of tuples:

```python
batch = [("h1", "c1"), ("h2", "c2"), ("h3", "c3")]
histories, currents = zip(*batch)
# histories = ("h1", "h2", "h3"), currents = ("c1", "c2", "c3")
```

### `lambda`: a tiny unnamed function

```python
double = lambda x: x * 2       # same as:  def double(x): return x * 2
```

### `slice` objects

`a[2:5]` takes items 2, 3, 4. You can store the `2:5` part in a variable with `slice`:

```python
s = slice(2, 5)
a[s]            # same as a[2:5]
```

That's handy when you'll use the same range on several arrays, as `history_for` does.

### `with open(...) as f:`

Opens a file and **guarantees it's closed** when the indented block ends, even if
an error happens. The mode says how: `"rb"` = read binary, `"wb"` = write binary.

### `try` / `except`

```python
try:
    risky_thing()
except RuntimeError as e:      # if risky_thing raises a RuntimeError, run this instead of crashing
    print("it failed:", e)
```

### `assert`

`assert condition` does nothing if the condition is True and **stops the program
with an error** if it's False. It's a built-in sanity check.

### Classes, `self`, `__init__`, inheritance, `super()`

A class is a blueprint for objects that bundle data and functions together.

```python
class Dog:
    def __init__(self, name):      # runs when you create a Dog: Dog("Rex")
        self.name = name           # self is "this particular dog"; store data on it
    def speak(self):
        return f"{self.name} says woof"

rex = Dog("Rex")
rex.speak()                        # -> "Rex says woof"
```

`class Tier0Model(nn.Module):` means Tier0Model **inherits** from PyTorch's
`nn.Module`: it gets all of Module's abilities (tracking weights, moving to the
GPU, …) for free. `super().__init__()` runs Module's own setup first; it's
required.

Methods with double underscores (`__init__`, `__len__`, `__getitem__`) are
special: Python calls them for you. `len(obj)` calls `obj.__len__()`, and
`obj[k]` calls `obj.__getitem__(k)`.

### Decorators (`@something` above a function)

```python
@torch.no_grad()
def predict(...):
```

A decorator wraps the function in extra behaviour. Here: "while this function
runs, PyTorch doesn't record anything for training." (See Part 3.)

### Generator expressions

Like a list comprehension but with round brackets; it produces items one by one:

```python
a, b, c = (x * 2 for x in (1, 2, 3))     # a=2, b=4, c=6
```

### Dict comprehensions

Like a list comprehension, but building a dict:

```python
{k: k * 2 for k in ["a", "b"]}     # -> {"a": "aa", "b": "bb"}
```

### Generators: functions that `yield`

A normal function `return`s one result and is finished. A function that uses
`yield` hands back results **one at a time**, pausing in between:

```python
def count_to(n):
    for i in range(1, n + 1):
        yield i            # hand back i, then pause here until the next one is asked for

for x in count_to(3):      # the for loop asks for values one by one
    print(x)               # 1, 2, 3
```

`batches()` in `tier0_train.py` works this way: each time the training loop asks
for the next batch, it builds exactly one and hands it over.

### `if __name__ == "__main__":`

When you run `python tier0_train.py`, Python sets `__name__` to `"__main__"`, so
`main()` runs. When another file does `import tier0_train`, `__name__` is
`"tier0_train"`, so `main()` does **not** run automatically. That lets the
functions be imported and tested without starting a training run.

### Paths with `pathlib`

```python
from pathlib import Path
p = Path("C:/data") / "processed" / "events.csv"    # / joins path pieces
```

---

## Part 2: NumPy and pandas, the bits used here

### NumPy arrays

A NumPy array is a grid of numbers, all the same type, that you can do maths on
all at once (no loops needed).

```python
import numpy as np
a = np.array([1, 2, 3])
a * 10              # -> [10, 20, 30]   (every element)
a.shape             # -> (3,)          one dimension, length 3
m = np.zeros((2, 3))
m.shape             # -> (2, 3)        2 rows, 3 columns
```

**Shape** is the single most important thing to track. `(n,)` is a list of n
numbers; `(n, 5)` is n rows of 5 numbers.

**dtype** is the number type: `float32` (decimals, 4 bytes each), `int64` (whole
numbers), `bool` (True/False). `.astype(np.float32)` converts.

### Indexing and slicing

```python
a = np.array([10, 20, 30, 40, 50])
a[1]       # -> 20
a[1:4]     # -> [20, 30, 40]   (start included, end excluded)
m[2]       # row 2 of a 2-D array
m[:, 0]    # column 0 of every row  (":" means "everything along this axis")
```

### Boolean masks

Comparing an array gives an array of True/False:

```python
gap = np.array([14, 6, 6, 4, 0])
gap >= 3                    # -> [True, True, True, True, False]
(gap >= 3) & (gap < 7)      # & = element-wise AND  -> [False, True, True, True, False]
```

Booleans act as 1 and 0 in arithmetic. **This is how the code hides values:**

```python
values = np.array([0.5, 0.2, 0.9])
known  = np.array([True, False, True])
values * known              # -> [0.5, 0.0, 0.9]   (multiplying by False gives 0)
```

### Adding an axis: `[:, None]`

`x[:, None]` turns shape `(n,)` into `(n, 1)`. Why? So it can multiply each row
of an `(n, 5)` array. NumPy then **broadcasts** (repeats) the single column
across all 5:

```python
known = np.array([True, False])         # shape (2,)
onehot = np.array([[0, 1, 0, 0, 0],
                   [1, 0, 0, 0, 0]])    # shape (2, 5)
onehot * known[:, None]                 # row 0 kept, row 1 zeroed
```

`x[None, :]` does the opposite: shape `(n,)` becomes `(1, n)`.

### Joining arrays

```python
np.concatenate([a, b])      # glue end to end:      (40,) + (23,) -> (63,)
np.stack([a, b], axis=1)    # put side by side as columns:  two (n,) -> (n, 2)
np.column_stack([A, b, c])  # 2-D arrays keep their columns, 1-D arrays become one column each
```

### Other NumPy functions used

| Function | What it does | Example |
|---|---|---|
| `np.log1p(x)` | log(1 + x), accurate even for tiny x | `np.log1p(0) = 0` |
| `np.sign(x)` | -1, 0 or +1 | `np.sign(-3) = -1` |
| `np.abs(x)` | absolute value | `np.abs(-3) = 3` |
| `np.sort(x)` | sorted copy | |
| `np.searchsorted(sorted, values)` | the position each value would sit at in the sorted array | `searchsorted([10,20,30], 20) = 1` |
| `np.arange(n)` | `[0, 1, …, n-1]` | |
| `np.flatnonzero(mask)` | positions where mask is True | `flatnonzero([F,T,T]) = [1,2]` |
| `np.bincount(ints)` | how many times each whole number appears | `bincount([0,2,2]) = [1,0,2]` |
| `.argmax()` | position of the largest value | `[1,0,2].argmax() = 2` |
| `np.datetime64("2025-01-01")` | a date NumPy can compare | |

### pandas: tables with column names

A **DataFrame** is a table; a **Series** is one column of it.

```python
import pandas as pd
df = pd.read_csv("events.csv")      # load a CSV into a DataFrame
df["ticker"]                        # one column (a Series)
df[["a", "b"]]                      # several columns (a smaller DataFrame)
df["x"].to_numpy()                  # hand the column over as a NumPy array
```

| Operation | What it does |
|---|---|
| `a.merge(b, on="key")` | Join two tables on a shared column, like VLOOKUP or a SQL join |
| `df.sort_values([...])` | Sort rows by one or more columns |
| `df.reset_index(drop=True)` | Renumber rows 0, 1, 2, … after sorting |
| `df.groupby("ticker")` | Treat each company's rows as its own group |
| `.cumcount()` | Within each group, number the rows 0, 1, 2, … |
| `.diff()` | Within each group, this row minus the previous row |
| `.fillna(0)` | Replace missing values with 0 |
| `.clip(-0.5, 0.5)` | Cap values: below -0.5 becomes -0.5, above 0.5 becomes 0.5 |
| `.notna()` | True where a value is present |
| `.map(f)` | Apply a function to every value |
| `pd.get_dummies(col)` | One 0/1 column per category value (see Part 6) |
| `pd.concat([...], axis=1)` | Glue tables side by side |

---

## Part 3: PyTorch from zero

### What is PyTorch for?

PyTorch does two jobs:

1. **Maths on big grids of numbers**, on the CPU or the GPU.
2. **Automatic differentiation**: it works out how to nudge every weight so the
   model gets a little less wrong. This is what makes training possible.

### Tensors

A **tensor** is PyTorch's version of a NumPy array. Same idea (a grid of numbers
with a shape and a dtype), plus two extras: it can live on the GPU, and it can
record how it was computed so gradients can flow back through it.

```python
import torch
t = torch.zeros(64, 63, 54)        # shape (64, 63, 54), all zeros
t = torch.from_numpy(numpy_array)  # NumPy -> tensor (shares the memory, no copy)
t = torch.tensor([1, 2, 3])        # Python list -> tensor
t.numpy()                          # tensor -> NumPy (must be on the CPU)
t.to("cuda")                       # copy to the GPU
t.cpu()                            # copy back to the CPU
t.item()                           # a 1-number tensor -> a plain Python number
```

**Device:** a tensor lives either on the CPU or on the GPU (`"cuda"`). **Every
tensor in one calculation must be on the same device.** That's why the training
code calls `.to(device)` on every batch.

**dtype:** `torch.float32` for numbers the model works with; `torch.int64` (also
called `long`) for class labels, which `CrossEntropyLoss` requires.

### Reading shapes like `(B, L, F)`

Most of the confusion in deep learning is about shapes, so the code always says
what they are. The letters used here:

- **B** = batch size: how many examples go through together (64)
- **L** = history length: the longest history in this batch
- **F** = features: how many numbers describe one thing (54 for a history row)
- **d_model** = the width of the model's internal vectors (64)

So `history` with shape `(B, L, F)` = `(64, 63, 54)` means: 64 examples, each
with up to 63 past events, each past event described by 54 numbers.

### Layers: `nn.Module`, `nn.Linear`

A **layer** is a small, reusable piece of computation with **learnable weights**.

`nn.Linear(in_features, out_features)` is the most basic one. It takes a vector
of `in_features` numbers and produces `out_features` numbers. Each output is a
weighted sum of all inputs plus a constant:

```
output_j = w_j1 * input_1 + w_j2 * input_2 + ... + w_jn * input_n + b_j
```

The `w`s and `b`s are the **weights** (or parameters). They start random, and
training adjusts them. `nn.Linear(54, 64)` has 54 × 64 = 3,456 weights plus 64
biases.

A Linear layer only acts on the **last** dimension, so it works on any shape:
`(64, 63, 54)` goes in, `(64, 63, 64)` comes out. It's applied to each of the
64 × 63 rows separately, with the same weights.

`nn.Module` is the base class for anything with weights. Your model is a Module
made of smaller Modules.

### Activation functions and dropout

Stacking only Linear layers is still just one big weighted sum. To learn curved,
non-linear patterns, you put an **activation** between them:

- **ReLU**: `max(0, x)`. Negative numbers become 0, positive ones pass through.
  Simple, and enough to make the network non-linear.

**Dropout(0.1)** randomly sets 10% of the numbers to zero during training (and
scales up the rest to compensate). It stops the model leaning too hard on any
single feature, which reduces **overfitting** (memorising the training data
instead of learning patterns). It's switched **off** during evaluation.

### How training works, in 6 steps

For each batch:

1. **Forward pass:** feed the inputs through the model and get predictions.
2. **Loss:** one number that says how wrong the predictions were.
3. **`optimizer.zero_grad()`:** clear the gradients left over from the last batch.
4. **`loss.backward()`:** PyTorch works out, for every weight, which direction
   would make the loss smaller and how strongly. That's the **gradient**.
5. **`optimizer.step()`:** nudge every weight a small step in its improving
   direction.
6. Repeat with the next batch.

One pass over all the training data is an **epoch**. We do 15.

**Learning rate** (`1e-3` = 0.001): how big each nudge is. Too big and training
jumps around; too small and it crawls.

**AdamW** is the optimizer, the rule for taking those steps. It adapts the step
size for each weight separately, based on how that weight's gradient has been
behaving. It's the standard default; you don't need its maths to use it.

### `model.train()` vs `model.eval()`, and `no_grad`

- `model.train()` switches dropout **on** (for training).
- `model.eval()` switches dropout **off** (for validation and testing).
- `torch.no_grad()` (or `torch.set_grad_enabled(False)`) tells PyTorch not to
  record anything for gradients. That's faster and uses less memory when you're
  only predicting.

### Loss functions used here

**SmoothL1Loss** (also called Huber loss) for the CAR prediction. With error
`e = prediction - truth`:

- if |e| < 1: loss = 0.5 × e²
- otherwise: loss = |e| - 0.5

Real values from PyTorch:

| prediction | truth | error | loss |
|---|---|---|---|
| 0.01 | 0.05 | 0.04 | 0.5 × 0.04² = **0.0008** |
| 0.00 | 3.00 | 3.00 | 3.0 - 0.5 = **2.5** |

Why not plain squared error? A wild 300% move (e = 3) would cost 0.5 × 9 = 4.5 and
pull the model hard towards outliers. Huber grows only in a straight line for big
errors, so a few extreme events can't take over training.

**CrossEntropyLoss** for the 5-class reaction. The model outputs 5 raw scores
called **logits**. Softmax turns them into probabilities that add to 1, and the
loss is `-log(probability given to the correct class)`. Real values:

```
logits         [ 2.0,  0.5,  0.1, -1.0, -2.0]
softmax        [0.694, 0.155, 0.104, 0.035, 0.013]   (adds up to 1)
if the truth is class 0: loss = -log(0.694) = 0.365   (confident and right: small loss)
if the truth is class 3: loss = -log(0.035) = 3.365   (confident and wrong: big loss)
```

### Dataset and DataLoader

- A **Dataset** answers two questions: "how many examples are there?"
  (`__len__`) and "give me example number k" (`__getitem__`).
- A **DataLoader** wraps a Dataset and serves **batches**: it asks for 64
  examples, then calls a **collate** function to stack them into tensors. With
  `shuffle=True` it uses a new random order every epoch.

**This project doesn't use them any more.** Building batches one example at a time
in Python kept the CPU busy (41 ms of CPU per batch). Instead, `tier0_train.py`
puts all the event data on the GPU once and builds each **whole batch** there with
a few tensor operations (`history_batch` and `batches`, Part 8). That uses 11.5 ms
of CPU per batch and gives identical batches. The idea is the same: a stream of
shuffled batches of 64.

---

## Part 4: Attention from zero

This is the heart of the model, so take it slowly.

### The problem attention solves

A company's history is a **list of past events of different lengths**: 0 for a
first filing, 128 for a busy company. The model needs to boil any such list down
to **one fixed-size summary**, and it should decide for itself **which past events
matter** for the current prediction. A big earnings surprise last quarter may
matter a lot; a routine governance filing three years ago probably doesn't.

The simplest summary would be a plain average of all past events. Attention is a
**weighted average where the model learns the weights**, and the weights change
depending on what's being asked.

### The analogy: a library search

- The **query** is what you're looking for ("recent big earnings reactions").
- Every past event has a **key**: a label saying what it's about.
- Every past event has a **value**: the actual information it carries.

You compare your query to every key. The better the match, the more of that
event's value you take. You end up with a blend of values weighted by
relevance.

### The maths, with real numbers

Say the query is `q = [1, 0]`, and there are three past events:

| event | key | value |
|---|---|---|
| 1 | [1, 0] | [10, 0] |
| 2 | [0, 1] | [0, 10] |
| 3 | [2, 0] | [5, 5] |

**Step 1: score each key by how well it matches the query.** The match measure
is the **dot product** (multiply matching positions and add up), divided by
√(size of the vectors) = √2 to keep the numbers in a sensible range:

```
event 1: (1×1 + 0×0) / √2 = 0.707
event 2: (1×0 + 0×1) / √2 = 0.000
event 3: (1×2 + 0×0) / √2 = 1.414
```

**Step 2: turn the scores into weights with softmax** (make them positive and add
up to 1; bigger score, bigger weight):

```
weights = [0.284, 0.140, 0.576]
```

Event 3 matched the query best, so it gets more than half the attention.

**Step 3: blend the values by those weights:**

```
output = 0.284 × [10, 0]  +  0.140 × [0, 10]  +  0.576 × [5, 5]
       = [5.72, 4.28]
```

That `[5.72, 4.28]` is the attention output: a summary of the three events,
leaning towards the ones relevant to the query. (These numbers were computed by
PyTorch.)

### Where do queries, keys and values come from?

They're **learned**. Each position's vector (64 numbers) is multiplied by three
separate learned weight matrices to produce its query, key and value. Training
adjusts those matrices, so the model learns **what to ask** (queries), **how to
label** events (keys) and **what information to pass on** (values). In PyTorch
these three matrices are stored stacked together as `in_proj_weight`, shape
(192, 64) = 3 × (64, 64).

### Self-attention: everyone attends to everyone

In **self**-attention, every position in the sequence makes its own query and
attends over all the positions. So each past event also looks at the other past
events. After one layer, each event's vector carries context about its
neighbours (for example, "this earnings filing came right after a CEO change").

### Masking: how attention ignores things

To make attention ignore a position (padding, in our case), set its score to
**minus infinity** before the softmax. Softmax turns minus infinity into a weight
of exactly 0. With event 3 masked out:

```
weights = [0.670, 0.330, 0.000]     output = [6.698, 3.302]
```

Event 3 contributes nothing; the other two share all the weight.

### Multi-head attention

Instead of one attention over all 64 numbers, the model runs **4 smaller
attentions in parallel** ("heads"), each on its own 16-number slice. Each head
can specialise: one might look for recent events, another for big reactions. The
4 results (16 numbers each) are glued back into 64 and mixed by one more Linear
layer (`out_proj`).

### A Transformer encoder layer = attention + a small network

One `nn.TransformerEncoderLayer` does this to every position:

```
x ──► self-attention (4 heads) ──► + x ──► LayerNorm ──► feed-forward ──► + ──► LayerNorm ──► out
 └────────────── "residual" ─────────┘                 └──────── residual ──────┘
```

- **Residual connection (`+ x`):** add the input back onto the output. The layer
  only has to learn a *change* to its input, which makes deep networks much
  easier to train.
- **LayerNorm:** rescale each position's 64 numbers to a standard range, which
  keeps training stable.
- **Feed-forward:** Linear(64→128) → ReLU → Linear(128→64), applied to each
  position separately. Attention mixes information *between* positions; the
  feed-forward network processes *within* each one.
- **Dropout** is applied inside as well.

We stack **2** such layers. Layer 2 attends over layer 1's outputs, so it works
with events that already carry context.

### The summary token (our query)

Here's the trick that produces one summary per company. We put an extra,
**learned** vector at the front of every history: the **summary token**. It isn't
a real event; it's a slot whose job is to collect a summary. Through
self-attention it reads every past event (with learned weights), and after the 2
layers we take **its** output as "the company's history in 64 numbers".

BERT, a famous language model, does exactly this with its `[CLS]` token. (A
second design, switched on with `CAUSAL_READOUT`, reads the summary from the last
past event instead. Part 7 explains both.) A bonus:
the summary token is never padding, so even a company's very first event (empty
history) has one valid position to attend to. Attention over zero items would
otherwise be undefined.

### What about time order?

A Transformer on its own treats its input as an unordered set. Language models
add "position encodings" so the model knows word order. Here we don't need them,
because every past event row already **contains its own timing**: how many
trading days ago it was, and how long after the previous filing it came. The
model can read time straight from the features. (`USE_TIME_CYCLES` optionally adds
more timing features; see Part 5.)

---

## Part 5: tier0_config.py, line by line

This file runs no logic. It's a list of settings that the other files read with
`import tier0_config as cfg`, then `cfg.SETTING_NAME`. Keeping them in one file
means you change a setting once and every script agrees.

By convention, names in ALL_CAPS are **constants**: values that are set once and
never changed while the program runs.

### Lines 1–7: docstring and import

```python
"""
tier0_config.py -- every setting the Tier 0 scripts share, in one place.
...
"""
from pathlib import Path
```

The docstring describes the file. `from pathlib import Path` brings in Python's
tool for building file paths that work on any operating system.

### Lines 9–14: where the files live

```python
DATA_DIR = Path(__file__).resolve().parents[1] / "data_phase2" / "processed"
```

Read it left to right:

1. `__file__`: a variable Python sets to the path of **this** file, e.g.
   `C:\...\DataMining\phase2\network\tier0_config.py`.
2. `Path(...)`: turn that text into a Path object.
3. `.resolve()`: make it a full, absolute path.
4. `.parents[1]`: go up the folder tree. `parents[0]` is the folder containing
   the file (`network`), and `parents[1]` is one level above (`phase2`).
5. `/ "data_phase2" / "processed"`: go down into `phase2\data_phase2\processed`.

**Why bother?** If the path were just `"data_phase2/processed"`, Python would
look for it relative to whichever folder you happened to run the script from, so
it would break depending on where you opened your terminal. Building it from the
file's own location makes it work from anywhere.

```python
COMPANIES_CSV = DATA_DIR / "companies.csv"
EVENTS_CSV = DATA_DIR / "events.csv"
MARKET_REACTIONS_CSV = DATA_DIR / "market_reactions.csv"
EVENTS_PKL = Path(__file__).resolve().parent / "tier0_events.pkl"   # written by prepare, read by train; kept beside the scripts
```

The full paths of the three input CSVs and the one output file. The CSVs stay in
`data_phase2/processed/` because other Phase 2 scripts share them; the pickle is used
only by the Tier 0 scripts, so it lives here in `network/`. Because both
scripts read `EVENTS_PKL` from here, the prepare script writes exactly where the
train script reads.

### Line 17: how much history

```python
MAX_HISTORY_LEN = 128      # the model sees at most this many of a company's most recent past events
```

Most companies have about 100 events, but some have over 1,000. Attention's cost
grows with the **square** of the sequence length, so one giant history would slow
down every batch it's in. The model sees each company's **most recent 128**
events.

### Lines 19–50: the leakage rule

This is the most important setting block. The long comment derives it; here it
is again, slowly.

**Setup.** We predict filing **B**'s CAR1, which adds up B's abnormal returns on
B's trading days -1, 0 and +1. Any earlier filing **A** whose reaction overlaps
those days would hand the model part of the answer.

**Measuring the gap.** Count trading days (Mon–Fri excluding market holidays)
from A's day 0 to B's day 0, and call that `gap`. Then A's day k falls on B's day
`(k - gap)`. Example: if gap = 5, A's day +1 is B's day -4.

**The rule.** A value of A may be shown only if the **last day it uses** falls
**before** B's day -1:

| A's value | last day it uses | condition | so we need |
|---|---|---|---|
| abnormal_return, volume change | A's day 0 | 0 - gap < -1 | gap ≥ 2 |
| CAR[-1,+1], reaction_category | A's day +1 | 1 - gap < -1 | gap ≥ 3 |
| CAR[-1,+5] | A's day +5 | 5 - gap < -1 | gap ≥ 7 |

`reaction_category` is just CAR[-1,+1] sorted into buckets, so it follows the
same rule as CAR[-1,+1].

**The worked example in the comment** (A on a Monday):

- B on **Wednesday** (gap 2): A's day 0 (Monday) is before B's day -1 (Tuesday),
  so A's abnormal return is shown. A's CAR[-1,+1] needs Tuesday, which *is* B's
  day -1, so it's hidden.
- B the **following Monday** (gap 5): A's CAR[-1,+1] ended on Tuesday, long before,
  so it's shown. A's CAR[-1,+5] runs to day +5, which is that following Monday:
  B's own day 0. Hidden until gap ≥ 7.
- B on the **same day** (gap 0): everything about A's reaction is hidden. They
  share one price move, so A's CAR1 **is** B's answer. (Part 9 shows this in your
  data.)

```python
MIN_GAP_DAY0 = 2
MIN_GAP_CAR1 = 3
MIN_GAP_CAR5 = 7
```

The three thresholds from the table, used in `history_batch` (Part 8).

**Hidden values and the `_known` flags.** A hidden value is replaced with 0. But
0 is also a real possible value (a stock that moved exactly 0%), so each hidden
group comes with a flag: 1 = this number is real, 0 = it was hidden. The model
can then tell "unknown" apart from "zero".

### Lines 52–60: the time-cycle experiment

```python
USE_TIME_CYCLES = False
TIME_CYCLE_PERIODS = (5, 21, 63, 252)   # trading days: week, month, quarter (earnings rhythm), year
```

Every past event always carries `log(1 + trading days ago)`: one number saying
how recent it was. With `USE_TIME_CYCLES = True`, each past event also gets, for
each period p, the pair:

```
sin(2π × days_ago / p)   and   cos(2π × days_ago / p)
```

**What that means.** Picture a clock hand that goes round once every p trading
days. The sin/cos pair is the hand's position (its x and y coordinates). Events
at the same point in the cycle get the same pair. For p = 63 (about a quarter),
an event 63 days ago and one 126 days ago land at the same spot, which makes it
easy for attention to notice "this happened at the same point in a previous
quarterly cycle". Why both sin **and** cos? With only one of them, two different
points on the clock can give the same value; the pair pins down the exact
position.

This is an **ablation** (an experiment that turns one ingredient off and on): run
training with False, then True, and compare the results.

### Lines 62–73: how the history is summarised (`CAUSAL_READOUT`)

```python
CAUSAL_READOUT = False
```

This chooses between two designs for turning the history into one summary. Both
see exactly the same past events, so leakage is identical either way; they differ
in *who may look at whom* inside attention, and *which position's output* is used.

- **False (default): bidirectional.** Every position may attend to every other
  position, and the summary token at position 0 is read out, as in BERT.
- **True: causal.** Each position may attend only to itself and **earlier**
  positions, like reading left to right. The **last real event's** position is
  read out, since it's the only one that has seen the whole history. Position 0 is
  then just a start token, read out only when there's no history at all.

Part 7 shows both in code. It's an A/B test: change only this flag, keep the same
seed and settings, and compare.

### Line 76: clipping inputs

```python
CLIP_RETURNS = 0.50        # past returns/CARs are clipped to +/-50% before use as INPUTS (targets stay raw)
```

Past returns used as **inputs** are capped at ±50%, so one freak +3,000% move
(some are data glitches) can't swamp everything else. The **target** (the answer
we train against) is never clipped.

### Line 84: the look-ahead switch

```python
DROP_LOOKAHEAD_FEATURES = False
```

The company profile is one snapshot from October 2026, attached even to 2016
events. The model therefore "knows" a 2016 company's 2026 size, which is
information from the future. The worst offender is `trailing_volatility_1y`: for
events in the test period, it was partly calculated from those events' own
returns. Setting this to True removes `volatility_tier` and
`trailing_volatility_1y` from the inputs. Run it both ways: if the results barely
change, the look-ahead isn't what's driving them.

### Lines 86–100: training settings

```python
TRAIN_END = "2025-01-01"   # events before this date train the model
VAL_END = "2025-07-01"     # events before this (and after TRAIN_END) are validation; the rest is test
```

The **time-based split**: the model learns from 2016–2024, is checked on the first
half of 2025, and is finally tested on July 2025 onwards. It always predicts
**later** periods than it learned from, the way it would have to in real life. A
random split would let it learn from 2026 and "predict" 2018, which is
unrealistically easy.

- **Train set**: the data the model learns from.
- **Validation set**: watched during training to spot overfitting (training loss
  keeps falling while validation loss rises means it's memorising).
- **Test set**: used **once**, at the end. It's the honest final score.

```python
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
```

64 examples per batch; 15 passes over the training data; step size 0.001
(Part 3).

```python
REACTION_LOSS_WEIGHT = 0.3
```

The total loss is `CAR loss + 0.3 × reaction-class loss`. CAR is the main task;
the class task is a helper and counts for less.

```python
TASK = "regression"
```

**What the main task is.** Two choices, everything else identical:

- `"regression"`: predict the CAR1 **value** itself (e.g. +0.0148), trained with
  Huber loss. MAE and R² measure it.
- `"direction"`: predict only **up or down** (is CAR1 above 0?). The model's single
  CAR output is then a **logit**: a score where above 0 means "up" and below 0
  means "down". It's trained with `BCEWithLogitsLoss` (binary cross-entropy: like
  the 5-class cross-entropy in Part 3, but for one yes/no question). Accuracy and
  per-side recall measure it.

The model itself doesn't change: it outputs one number either way. Only the
target, the loss and the evaluation change (Part 8).

```python
SEED = 0
```

Training involves randomness: the starting weights, the shuffle order, dropout.
A **seed** makes that randomness repeat exactly, so two runs with the same seed
and settings give the same result. For an experiment, run each side with seeds 0,
1 and 2. If the difference between settings is smaller than the difference between
seeds, it's noise.

---

## Part 6: tier0_prepare_data.py, line by line

**Job:** read the three CSVs and save **one row of numbers per event**, in three
groups:

- **event_x**: *what was filed*. Known as soon as the filing exists. (40 numbers)
- **company_x**: *the company's profile*. The same for all of a company's events.
  (23 numbers)
- **outcome**: *how the market reacted*. The model only sees these for **past**
  events, and only once their window has closed (`history_for`, Part 8).

Plus the **targets** (the answers) and some bookkeeping columns. It does **not**
build histories; training builds those on the fly.

### Lines 18–23: imports

```python
import pickle

import numpy as np
import pandas as pd

import tier0_config as cfg
```

- `pickle`: Python's way to save almost any object to a file and load it back
  later, exactly as it was.
- `numpy as np`, `pandas as pd`: the usual short names.
- `import tier0_config as cfg`: our settings file, available as `cfg`.

### Line 25: the five reaction classes

```python
REACTION_CLASSES = ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]
```

The order matters: a label's **position** in this list becomes its class number
(Strong Negative = 0, … Strong Positive = 4).

### Lines 28–30: `signed_log`

```python
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))
```

It takes the absolute value, applies `log(1 + |x|)`, then puts the sign back.

| x | signed_log(x) |
|---|---|
| 0.05 (+5%) | 0.0488 |
| -0.05 | -0.0488 |
| 0.5 | 0.405 |
| 3.0 (+300%) | 1.386 |

Small values barely change; huge values are squashed. This is the "fat tails"
problem: financial returns have rare but enormous outliers, and neural networks
learn better when inputs sit in a moderate range. The sign is kept, so up stays up
and down stays down. Used for **inputs** only.

### Lines 33–35: `one_hot`

```python
def one_hot(series, prefix):
    return pd.get_dummies(series, prefix=prefix, prefix_sep="=").astype(np.float32)
```

A model needs numbers, not words. **One-hot encoding** turns a category column
into one 0/1 column per possible value:

```
event_category                 event_category=Financial/Earnings  event_category=Other  ...
"Financial/Earnings"   ──►                  1                              0
"Other"                                     0                              1
```

- `pd.get_dummies` does the work (a "dummy" column = a 0/1 indicator).
- `prefix=prefix, prefix_sep="="` names the columns `prefix=value`, which is
  readable later.
- `.astype(np.float32)`: get_dummies gives True/False; this turns them into
  1.0/0.0.
- A missing value gets **all zeros** (no column set).

Why not just number the categories 1, 2, 3…? That would tell the model "Other"
(3) is somehow bigger than "Earnings" (1), which is meaningless. One-hot keeps
every category separate and equal.

### Line 38: `main()`

```python
def main():
```

All the work happens inside `main()`, which runs only when the file is executed
directly (lines 129–130).

### Line 40: reading options

```python
read = dict(keep_default_na=False, na_values=[""])   # a real ticker "NA" must stay text
```

By default, pandas treats text like `"NA"`, `"N/A"` and `"NULL"` as **missing**. A
company whose ticker really is `NA` would vanish. These two options say: don't use
the default list (`keep_default_na=False`); only a truly empty cell counts as
missing (`na_values=[""]`).

### Lines 41–43: load the CSVs

```python
companies = pd.read_csv(cfg.COMPANIES_CSV, **read)
events = pd.read_csv(cfg.EVENTS_CSV, parse_dates=["filing_date", "report_date"], **read)
reactions = pd.read_csv(cfg.MARKET_REACTIONS_CSV, parse_dates=["trading_date"], **read)
```

- `**read` spreads the options from line 40 into each call (Part 1).
- `parse_dates=[...]` turns those text columns (`"2016-01-25"`) into real dates,
  so we can subtract them and compare them.

### Lines 45–46: join the tables

```python
df = events.merge(reactions, on="event_id", validate="one_to_one")
df = df.merge(companies, on="ticker", how="left", validate="many_to_one")
```

- Line 45 lines up each event with its market reaction, matched by `event_id`.
  `validate="one_to_one"` makes pandas **raise an error** unless each event_id
  appears exactly once on each side. A mistake there would silently duplicate
  rows, so this is a safety check.
- Line 46 attaches the company profile to every event, matched by `ticker`.
  `"many_to_one"`: many events can share one company, but each ticker must appear
  once in companies.csv. `how="left"` keeps every event even if its company were
  missing (none are).

`df` now has one row per event, with every column from all three files.

### Line 49: sanity check

```python
assert df[["abnormal_return", "car_minus1_plus1", "car_minus1_plus5"]].notna().all().all()
```

Read it inside out:

1. `df[[...]]`: the three return columns.
2. `.notna()`: True wherever a value exists.
3. `.all()`: for each column, "are all values present?" That gives 3 True/False
   values.
4. `.all()` again: "are all 3 True?" That gives one True/False.
5. `assert`: stop with an error if it's False.

The Phase 2 data pipeline already dropped every event without a full price
window, so this should always pass. Checking costs one line; a silent missing
value would poison training.

### Line 53: sort

```python
df = df.sort_values(["ticker", "report_date", "event_id"]).reset_index(drop=True)
```

- Sort by company, then date, then event_id. Each company's events are now
  together and in time order. When a company filed twice on the same day,
  `event_id` decides their order the same way every run.
- `.reset_index(drop=True)` renumbers the rows 0, 1, 2, …. **This matters a
  lot:** from now on, a row's number *is* its position, and `history_for` relies
  on "the rows just before this one belong to the same company, in time order".

### Lines 54–55: first events and company start rows

```python
first_row = df.groupby("ticker").cumcount().to_numpy() == 0
company_start = np.arange(len(df)) - df.groupby("ticker").cumcount().to_numpy()
```

`df.groupby("ticker").cumcount()` numbers each company's rows 0, 1, 2, …. Say
company A has rows 0–2 and company B has rows 3–6:

```
row number:     0  1  2  3  4  5  6
ticker:         A  A  A  B  B  B  B
cumcount:       0  1  2  0  1  2  3
first_row:      T  F  F  T  F  F  F      (cumcount == 0)
company_start:  0  0  0  3  3  3  3      (row number - cumcount)
```

- `first_row`: is this the company's very first event?
- `company_start`: the row where this company's events begin. `history_for` uses
  it so a history **never runs into the previous company's events**.

### Lines 60–61: trading-day numbers

```python
calendar = np.sort(df["trading_date"].unique())
trading_day = np.searchsorted(calendar, df["trading_date"].to_numpy())
```

Each event has a `trading_date`: its day 0, the first trading day on or after the
report date.

1. `calendar`: every distinct trading date in the data, sorted. With 411k events
   over 10 years, every trading day appears. That's 2,702 days.
2. `np.searchsorted(calendar, dates)`: each date's position in that calendar.
   The first trading day of 2016 is day 0, the next is day 1, and so on.

Now "how many trading days apart are two events?" is just **subtraction** of
their numbers. Weekends and holidays never count, because they aren't in the
calendar. That's exactly what the leakage rule needs.

### Line 64: days since the previous filing

```python
days_since_prev = df.groupby("ticker")["report_date"].diff().dt.days.fillna(0).to_numpy()
```

1. `df.groupby("ticker")["report_date"]`: each company's report dates.
2. `.diff()`: each date minus the previous one **within the same company** (a time
   span). Each company's first event has nothing before it, so it gets NaN
   (missing).
3. `.dt.days`: turn the time span into a number of days.
4. `.fillna(0)`: the first event gets 0.
5. `.to_numpy()`: as a NumPy array.

These are calendar days, so they measure how busy the company's filing schedule
is. The leakage rule uses trading days instead.

### Lines 65–74: building `event_x` (what was filed)

```python
event_x = pd.concat([
    one_hot(df["event_category"], "event_category"),
    one_hot(df["filing_timeliness"], "filing_timeliness"),
    df["item_codes"].fillna("").str.get_dummies(sep=",").add_prefix("item=").astype(np.float32),
    pd.DataFrame({
        "num_items_disclosed": df["num_items_disclosed"].astype(np.float32),
        "log_days_since_prev_filing": np.log1p(days_since_prev).astype(np.float32),
        "is_first_event": first_row.astype(np.float32),
    }),
], axis=1)
```

`pd.concat([...], axis=1)` glues four blocks of columns side by side
(`axis=1` = add columns; `axis=0` would add rows):

1. **event_category** one-hot: 7 columns (Capital/Financing, Financial/Earnings,
   Legal/Regulatory, M&A/Contracts, Management/Governance, Other, Security
   Incident).
2. **filing_timeliness** one-hot: 3 columns (Late, On-time, Slightly Late).
3. **item codes**, multi-hot: 27 columns (`item=1.01` … `item=9.01`). A filing can
   list several items, like `"2.02,9.01"`, so it gets a 1 in each of those
   columns. That's "multi-hot": possibly several 1s per row.
   - `.fillna("")`: a missing value becomes an empty string.
   - `.str.get_dummies(sep=",")`: split each string on commas and make a 0/1
     column per distinct code.
   - `.add_prefix("item=")`: the column `2.02` becomes `item=2.02`.
4. **three numbers**:
   - `num_items_disclosed`: how many items the filing lists.
   - `log_days_since_prev_filing`: log(1 + days); the log squashes long gaps.
   - `is_first_event`: 1.0 for the company's first event, otherwise 0.0. This
     separates "first ever" (days_since_prev = 0 because there's no previous
     filing) from "same day as the last filing" (days_since_prev = 0 for real).

Total: 7 + 3 + 27 + 3 = **40 columns**.

### Lines 77–78: a helper for money-sized numbers

```python
def log_or_zero(col):
    return np.log1p(df[col].fillna(0).clip(lower=0)).astype(np.float32)
```

A small function defined inside `main()` so it can use `df`. For a column name:

1. `df[col].fillna(0)`: missing becomes 0 (a few companies lack revenue or
   employee counts).
2. `.clip(lower=0)`: no negatives (protects the log).
3. `np.log1p(...)`: market caps run from $32 thousand to $5.6 trillion. Raw, the
   big companies would drown everything; after log(1 + x) the range is about
   10.4 to 29.4.

### Lines 80–90: building `company_x` (the profile)

```python
profile = [
    one_hot(df["sector"], "sector"),
    one_hot(df["size_category"], "size_category"),
    pd.DataFrame({"log_market_cap": log_or_zero("market_cap"),
                  "log_revenue_ttm": log_or_zero("revenue_ttm"),
                  "log_employees": log_or_zero("employees")}),
]
if not cfg.DROP_LOOKAHEAD_FEATURES:
    profile += [one_hot(df["volatility_tier"], "volatility_tier"),
                pd.DataFrame({"log_trailing_volatility_1y": log_or_zero("trailing_volatility_1y")})]
company_x = pd.concat(profile, axis=1)
```

- `profile` starts as a list of three blocks: sector (11 columns), size category
  (4) and three log numbers (3).
- `if not cfg.DROP_LOOKAHEAD_FEATURES:` unless the look-ahead switch is on,
  `profile += [...]` appends two more blocks: volatility tier (4) and log trailing
  volatility (1). (`+=` on a list adds the new items to the end.)
- `pd.concat(profile, axis=1)` glues the blocks side by side.

Total: 11 + 4 + 3 + 4 + 1 = **23 columns** (or 18 with the switch on).

### Lines 93–103: the outcome values (how the market reacted)

```python
clip = cfg.CLIP_RETURNS
volume = df["volume_change_pct"]
outcome = {
    "abnormal_return": signed_log(df["abnormal_return"].clip(-clip, clip)).to_numpy(np.float32),
    "car1": signed_log(df["car_minus1_plus1"].clip(-clip, clip)).to_numpy(np.float32),
    "car5": signed_log(df["car_minus1_plus5"].clip(-clip, clip)).to_numpy(np.float32),
    "volume_change": signed_log(volume.fillna(0)).to_numpy(np.float32),
    "volume_present": volume.notna().to_numpy(),
    "reaction_onehot": np.stack([(df["reaction_category"] == c).to_numpy(np.float32)
                                 for c in REACTION_CLASSES], axis=1),
}
```

A dict of arrays, one value per event, prepared for use as **history inputs**:

- **abnormal_return, car1, car5**: clip to ±50%, then `signed_log`, then a float32
  array.
- **volume_change**: how unusual the event-day trading volume was, signed-logged.
  A few events have no value (their baseline volume was zero), so `fillna(0)`...
- **volume_present**: ...and this True/False array remembers which ones were
  real.
- **reaction_onehot**: the reaction category as 5 columns. Reading the inner part:
  - `for c in REACTION_CLASSES`: loop over the 5 class names.
  - `(df["reaction_category"] == c)`: True where the event's category is `c`.
  - `.to_numpy(np.float32)`: becomes 1.0/0.0.
  - The list comprehension makes 5 such arrays; `np.stack(..., axis=1)` puts them
    side by side as columns, giving shape (events, 5).

(This builds the one-hot by hand rather than with `one_hot()` so that the column
order is guaranteed to follow `REACTION_CLASSES`.)

### Lines 106–107: the targets (the answers)

```python
target_car1 = df["car_minus1_plus1"].to_numpy(np.float32)
target_class = df["reaction_category"].map(REACTION_CLASSES.index).to_numpy(np.int64)
```

- `target_car1`: the real CAR1, **not clipped or logged**. This is what the model
  learns to predict.
- `target_class`: `.map(REACTION_CLASSES.index)` replaces each label with its
  position in the list, so "Negative" becomes 1. `REACTION_CLASSES.index` is the
  list's own "find the position of" function, passed in as the thing to apply.
  It's int64 because CrossEntropyLoss requires whole-number labels.

### Lines 109–118: assemble everything into one dict

```python
data = {
    "event_x": event_x.to_numpy(np.float32), "event_columns": list(event_x.columns),
    "company_x": company_x.to_numpy(np.float32), "company_columns": list(company_x.columns),
    **outcome,
    "reaction_classes": REACTION_CLASSES,
    "target_car1": target_car1, "target_class": target_class,
    "trading_day": trading_day, "company_start": company_start,
    "event_date": df["report_date"].to_numpy(), "ticker": df["ticker"].to_numpy(),
    "event_id": df["event_id"].to_numpy(),
}
```

- The tables become plain NumPy number grids (fast to slice), but the **column
  names are saved alongside**, so you can always tell what column 17 means.
- `**outcome` copies all six outcome arrays into this dict (Part 1).
- `trading_day` and `company_start` are what `history_for` needs.
- `event_date` drives the train/validation/test split; `ticker` and `event_id` are
  for looking things up later.

Every array has **one entry per event, in the same sorted order**, so row `i` of
any array refers to the same event.

### Lines 119–120: save

```python
with open(cfg.EVENTS_PKL, "wb") as f:
    pickle.dump(data, f)
```

Open the output file for **w**riting in **b**inary mode, and pickle the whole
dict into it. About 141 MB.

### Lines 122–126: report

```python
print(f"{len(df):,} events from {df['ticker'].nunique():,} companies, "
      f"{len(calendar):,} trading days ({str(calendar[0])[:10]} to {str(calendar[-1])[:10]})")
```

Prints a summary like `411,685 events from 3,900 companies, 2,702 trading days
(2016-01-04 to 2026-10-01)`.

- `df['ticker'].nunique()`: the number of distinct tickers.
- `str(calendar[0])[:10]`: the first date as text, keeping the first 10
  characters (the `YYYY-MM-DD` part).
- `calendar[-1]`: index -1 means the **last** element.

```python
print(f"event_x: {event_x.shape[1]} columns | company_x: {company_x.shape[1]} columns"
      f"{' (look-ahead features dropped)' if cfg.DROP_LOOKAHEAD_FEATURES else ''}")
```

`.shape[1]` is the number of columns. The second f-string adds a note only when
the switch is on: an inline if, returning either the note or an empty string.

```python
print(f"saved {cfg.EVENTS_PKL} ({cfg.EVENTS_PKL.stat().st_size / 1e6:.0f} MB)")
```

`.stat().st_size` is the file size in bytes; divided by `1e6` (one million) that's
megabytes, and `:.0f` means no decimals.

### Lines 129–130

```python
if __name__ == "__main__":
    main()
```

Run `main()` only when this file is executed directly (Part 1).

---

## Part 7: tier0_model.py, line by line

**Job:** define the network. This file doesn't load data or train anything; it
only describes what happens to the numbers. Read Part 4 (attention) first if you
haven't.

### The whole model in one picture

```
 history (B, L, 54)                                   current (B, 63)
      │                                                     │
 history_proj: Linear 54→64                           current_proj: Linear 63→64
      │                                                     │
 (B, L, 64)                                            (B, 64)
      │                                                     │
 put the summary token in front → (B, L+1, 64)             │
      │                                                     │
 Transformer encoder (2 layers, 4 heads, padding masked)   │
      │                                                     │
 take the summary → (B, 64)   (see Lines 55–65)            │
      │                                                     │
      └──────────────── glue side by side ─────────────────┘
                              │
                          (B, 128)
                              │
               combine: Linear 128→64, ReLU, Dropout
                              │
                          (B, 64)
                       ┌──────┴──────┐
              car_head: Linear 64→1   class_head: Linear 64→5
                       │              │
             predicted CAR1 (B,)   5 class scores (B, 5)
```

### Lines 1–27: docstring and imports

```python
import torch
import torch.nn as nn
```

`torch` is PyTorch. `torch.nn` ("neural network") holds the building blocks
(layers, losses); `as nn` is the usual short name.

### Line 30: the class

```python
class Tier0Model(nn.Module):
```

Our model is a class that **inherits from `nn.Module`** (Part 1). That gives it,
for free:

- tracking of every learnable weight inside it (`model.parameters()`);
- moving everything to the GPU with one call (`model.to(device)`);
- train/eval switching (`model.train()`, `model.eval()`);
- making the object callable: `model(x)` runs `forward(x)`.

### Lines 31–34: the constructor

```python
def __init__(self, n_history_features, n_current_features, n_classes,
             d_model=64, nhead=4, num_layers=2, dropout=0.1, causal_readout=False):
    super().__init__()
    self.causal_readout = causal_readout
```

`__init__` runs **once**, when the model is created. Its job is to build all the
layers. The inputs:

| argument | value | meaning |
|---|---|---|
| `n_history_features` | 54 (62 with cycles) | numbers per past-event row |
| `n_current_features` | 63 | numbers describing the current filing |
| `n_classes` | 5 | reaction categories |
| `d_model=64` | 64 | width of every internal vector |
| `nhead=4` | 4 | attention heads (64 / 4 = 16 numbers per head) |
| `num_layers=2` | 2 | stacked encoder layers |
| `dropout=0.1` | 0.1 | fraction zeroed during training |
| `causal_readout=False` | from `CAUSAL_READOUT` | which summary design to use (Lines 55–65) |

The `=64` etc. are **default values**: used unless the caller passes something
else. `tier0_train.py` passes only the first three, so the defaults apply.

`super().__init__()` runs `nn.Module`'s own setup. Forgetting it causes an error
as soon as you add a layer. `self.causal_readout = causal_readout` stores the
design choice so `forward` can check it later.

### Lines 35–36: the two input projections

```python
self.history_proj = nn.Linear(n_history_features, d_model)
self.current_proj = nn.Linear(n_current_features, d_model)
```

The raw inputs have different sizes (54 and 63) and mix very different things
(0/1 flags, logs, returns). Each Linear layer learns to translate its input into
the model's common 64-number "language".

- `history_proj` is applied to **every past event row** separately, with the same
  weights for all of them. So "an earnings filing with a +5% reaction" is
  translated the same way wherever it sits in a history.
- `self.` stores the layer on the model, so PyTorch tracks its weights.

### Line 37: the summary token

```python
self.summary_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
```

- `torch.randn(1, 1, 64)`: 64 random numbers from a bell curve, shaped
  (1 example, 1 position, 64 numbers).
- `* 0.02`: make them small, so the token starts near zero.
- `nn.Parameter(...)`: **mark this tensor as learnable**. Training will adjust
  these 64 numbers like any other weight. It learns what a good "question" to ask
  the history is (Part 4).

### Lines 38–40: the Transformer encoder

```python
layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2 * d_model,
                                   dropout=dropout, batch_first=True)
self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
```

**Line 38** builds **one** encoder layer: self-attention, then a small
feed-forward network, each with a residual connection and LayerNorm (Part 4's
diagram).

- `d_model` (64) is the vector width going in and out.
- `nhead` (4) is the number of attention heads.
- `dim_feedforward=2 * d_model` (128) is the hidden width inside the feed-forward
  part: 64 → 128 → 64.
- `dropout=dropout` applies 10% dropout inside the layer.
- `batch_first=True` means our tensors are shaped **(batch, sequence, features)**.
  PyTorch's old default was (sequence, batch, features); this order is easier to
  read.

**Line 40** stacks `num_layers` (2) **copies** of that layer. Each copy gets its
own separate weights; they don't share. `enable_nested_tensor=False` turns off a
PyTorch speed trick for padded batches. It doesn't change any results; with it
on, PyTorch prints warnings about our mask setup, so it's switched off to keep
things simple and quiet.

### Line 41: the combiner

```python
self.combine = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
```

`nn.Sequential` chains layers so that `self.combine(x)` runs them in order:

1. `nn.Linear(128, 64)`: takes [history summary (64) + current filing (64)] and
   mixes them into 64 numbers. This is where the model can learn interactions
   like "an earnings filing **at a company whose last earnings dropped 10%**".
2. `nn.ReLU()`: the non-linearity (Part 3).
3. `nn.Dropout(0.1)`: regularisation (Part 3).

### Lines 42–43: the two output heads

```python
self.car_head = nn.Linear(d_model, 1)
self.class_head = nn.Linear(d_model, n_classes)
```

Both read the same 64-number combined vector:

- `car_head`: 64 → **1 number**: the predicted CAR1, or, with `TASK = "direction"`,
  a logit for "up" (above 0 = up).
- `class_head`: 64 → **5 numbers**, a score (logit) for each reaction class.
  Softmax would turn these into probabilities; `CrossEntropyLoss` does that
  internally (Part 3).

### Where all 83,270 weights are

| part | shape | count |
|---|---|---|
| `summary_token` | (1, 1, 64) | 64 |
| `history_proj` | 64×54 weights + 64 biases | 3,520 |
| `current_proj` | 64×63 + 64 | 4,096 |
| **each** encoder layer: attention Q/K/V (`in_proj`) | 192×64 + 192 | 12,480 |
| …attention output (`out_proj`) | 64×64 + 64 | 4,160 |
| …feed-forward 64→128 (`linear1`) | 128×64 + 128 | 8,320 |
| …feed-forward 128→64 (`linear2`) | 64×128 + 64 | 8,256 |
| …two LayerNorms | 4 × 64 | 256 |
| encoder layers × 2 | | 66,944 |
| `combine` | 64×128 + 64 | 8,256 |
| `car_head` | 1×64 + 1 | 65 |
| `class_head` | 5×64 + 5 | 325 |
| **total** | | **83,270** |

That's small by deep-learning standards (language models have billions), which is
right for about 330k training examples. More weights would mostly memorise.

### Lines 45–47: `forward`, where the computation happens

```python
def forward(self, history, history_len, current):
```

`forward` runs every time the model is called: `model(history, history_len,
current)`. Its inputs, for a batch of B examples:

- `history`: (B, L, 54). Each example's past events; shorter histories are
  padded with zero rows up to L, the longest in the batch.
- `history_len`: (B,). The **real** history length of each example, so the
  model knows which rows are padding.
- `current`: (B, 63). Each example's current filing.

### Line 48: read the shape

```python
B, L, _ = history.shape
```

`history.shape` is a tuple like `(64, 63, 54)`. Unpacking gives `B = 64`,
`L = 63`, and the 54 goes into `_` because we don't need it.

### Line 51: build the sequence

```python
seq = torch.cat([self.summary_token.expand(B, 1, -1), self.history_proj(history)], dim=1)
```

Inside out:

1. `self.history_proj(history)`: (B, L, 54) → **(B, L, 64)**. Every past-event row
   is translated to 64 numbers.
2. `self.summary_token.expand(B, 1, -1)`: the token is (1, 1, 64); `expand`
   repeats it B times, giving **(B, 1, 64)**: one copy at the front of each
   example. `-1` means "keep this dimension's size". `expand` doesn't copy any
   memory; all B copies are views of the same 64 learnable numbers.
3. `torch.cat([...], dim=1)`: glue along dimension 1 (the sequence dimension):
   (B, 1, 64) + (B, L, 64) → **(B, L+1, 64)**.

Position 0 of every sequence is now the summary token, and positions 1…L are the
past events, oldest first.

### Lines 52–53: mark the padding

```python
position = torch.arange(L + 1, device=history.device)
is_padding = position[None, :] > history_len[:, None]      # position 0 (summary) is never padding
```

- `torch.arange(L + 1)` gives `[0, 1, 2, …, L]`: the position numbers.
  `device=history.device` creates it on the same device (CPU or GPU) as the data,
  because tensors on different devices can't be combined.
- `position[None, :]` has shape (1, L+1): one row of positions.
- `history_len[:, None]` has shape (B, 1): one column of lengths.
- Comparing a row with a column **broadcasts** to a full grid, (B, L+1).

Example with L = 4 and two examples whose real lengths are 2 and 4:

```
position:              0      1      2      3      4
length 2:  is_padding  False  False  False  True   True     ← positions 3,4 are padding
length 4:  is_padding  False  False  False  False  False    ← no padding
```

A position is padding if it's **after** the real events. The summary sits at
position 0, and 0 > length is never true, so it's never padding. Even with
length 0, the summary stays visible, which is why an empty history still works.

### Lines 55–65: run the encoder and take the summary (two designs)

```python
if not self.causal_readout:
    summary = self.encoder(seq, src_key_padding_mask=is_padding)[:, 0]
else:
    looks_ahead = torch.triu(torch.ones(L + 1, L + 1, dtype=torch.bool, device=history.device), diagonal=1)
    encoded = self.encoder(seq, mask=looks_ahead, src_key_padding_mask=is_padding)
    summary = encoded[torch.arange(B, device=history.device), history_len]
```

`self.causal_readout` (from `CAUSAL_READOUT` in the config) picks one of two
designs. Both see the same past events, and neither can leak, because the history
only ever contains the past. They differ in **who may look at whom** inside
attention, and **which position's output** becomes the summary.

#### Design A (`CAUSAL_READOUT = False`, the default): look at everything, read the summary token

```python
summary = self.encoder(seq, src_key_padding_mask=is_padding)[:, 0]
```

- `self.encoder(seq, src_key_padding_mask=is_padding)`: runs the 2 Transformer
  layers. `src_key_padding_mask` says which positions to ignore (True = ignore).
  Inside attention, those positions get a score of minus infinity, so their weight
  is exactly 0 (Part 4's masking example). Output: **(B, L+1, 64)**, an updated
  vector for every position.
- `[:, 0]`: for every example (`:`), take position 0, the summary token. Result:
  **(B, 64)**: each company's history boiled down to 64 numbers.

The other positions' outputs are thrown away. They were still useful: in layer 1
the past events attended to each other, so in layer 2 the summary reads events
that already carry context.

#### Design B (`CAUSAL_READOUT = True`): look only backwards, read the last real event

**The causal mask.** First, a grid of who may look at whom:

```python
looks_ahead = torch.triu(torch.ones(L + 1, L + 1, dtype=torch.bool, device=history.device), diagonal=1)
```

- `torch.ones(L+1, L+1, dtype=torch.bool)`: a square grid of True, one row and
  one column per position.
- `torch.triu(..., diagonal=1)`: "upper triangle". It keeps only the entries
  strictly above the diagonal and sets everything else to False.

Rows are the position **doing the looking**, columns the position **being looked
at**, and True means **blocked**. With 4 positions:

```
                 looked at →   0       1       2       3
looking  0  (start token)     False   True    True    True
         1  (event 1)         False   False   True    True
         2  (event 2)         False   False   False   True
         3  (event 3)         False   False   False   False
```

Everything above the diagonal is "looking at a later position", and it's all
blocked. So event 2 can see the start token, event 1 and itself, but not event 3.
Each event only ever sees itself and what came before it.

```python
encoded = self.encoder(seq, mask=looks_ahead, src_key_padding_mask=is_padding)
```

`mask=looks_ahead` applies that grid in every layer, and the padding mask still
applies on top. Output: **(B, L+1, 64)**.

**The readout.**

```python
summary = encoded[torch.arange(B, device=history.device), history_len]
```

This is "advanced indexing": take two index lists of the same length and pick one
element per pair. `torch.arange(B)` is `[0, 1, …, B-1]` (which example) and
`history_len` is, for example, `[0, 5, 2, …]` (which position). So it takes
example 0's position 0, example 1's position 5, example 2's position 2, and so on.
Result: **(B, 64)**.

**Why position `history_len`?** The real events sit at positions 1 … history_len
(the start token is at 0), so the **last real event** is at position
`history_len`. In general terms that's `true_len - 1`, where
`true_len = history_len + 1` counts the start token.

**Why read the last one?** With the causal mask, earlier positions can't see later
ones, so the **last** real position is the only one that has seen the whole history
(including the start token). It's like reading a story left to right and asking
the reader for a summary on the last page.

**And with no history?** If `history_len` is 0, the readout is position 0, the
start token. That's why it's kept: there's always one valid position to read,
even for a company's very first event.

This was verified on real data: the model's output equals a readout at position
`history_len` for every example, equals a position-0 readout only for the
empty-history example, and with the mask, an event's output is identical whether
or not later events exist.

#### Which design is better?

Nobody knows yet for this data; that's what the A/B test is for. Design A lets the
summary token weigh all events symmetrically. Design B builds in a sense of order:
each event is understood in light of what came before it, and the most recent
event is the readout point. That can matter for event sequences.

### Line 68: combine with the current filing

```python
hidden = self.combine(torch.cat([summary, self.current_proj(current)], dim=-1))
```

1. `self.current_proj(current)`: (B, 63) → (B, 64).
2. `torch.cat([summary, ...], dim=-1)`: glue side by side along the **last**
   dimension (`-1` = last): (B, 64) + (B, 64) → **(B, 128)**.
3. `self.combine(...)`: Linear 128→64, ReLU, Dropout → **(B, 64)**.

### Line 69: the two predictions

```python
return self.car_head(hidden).squeeze(-1), self.class_head(hidden)
```

- `self.car_head(hidden)`: (B, 64) → (B, 1). `.squeeze(-1)` removes that size-1
  last dimension, giving **(B,)**: one predicted CAR per example, the same shape
  as the targets.
- `self.class_head(hidden)`: (B, 64) → **(B, 5)** class scores.
- `return a, b` returns both as a pair; the caller unpacks them with
  `car_pred, logits = model(...)`.

---

## Part 8: tier0_train.py, line by line

**Job:** build each example's history (with the leakage rule), feed batches to
the model, train it, and compare it with a naive baseline on the test period.

### Lines 18–25: imports

```python
import pickle

import numpy as np
import torch
import torch.nn as nn

import tier0_config as cfg
from tier0_model import Tier0Model
```

- `from tier0_model import Tier0Model`: our network class from Part 7.
- There's no `DataLoader` import: batches are built on the GPU instead (Part 3).
- There's no scikit-learn here on purpose. The three metrics are one line of
  NumPy each, and importing scikit-learn after PyTorch crashed Python on this
  Windows machine.

### Lines 27–34: `history_columns`, the name of every history column

```python
def history_columns(d):
    cycles = [f"{f}(2pi*days_ago/{p})" for p in cfg.TIME_CYCLE_PERIODS for f in ("sin", "cos")]
    return (d["event_columns"]
            + ["log_trading_days_ago"] + (cycles if cfg.USE_TIME_CYCLES else [])
            + ["abnormal_return", "volume_change", "car1", "car5"]
            + [f"reaction={c}" for c in d["reaction_classes"]]
            + ["day0_known", "volume_known", "car1_known", "car5_known"])
```

It returns a list of names, one per column of a history row, **in the same order
as `history_batch` builds them**:

- `cycles`: a nested list comprehension (Part 1). For each period p, `"sin(...)"`
  then `"cos(...)"`, giving 8 names. Used only if cycles are switched on: the
  inline if adds them or an empty list.
- `+` between lists glues them together.

It's used for two things: to tell the model how many history features there are
(`len(...)` → 54), and so that you can label columns when inspecting a history.
The full list, with cycles off:

| columns | count | what |
|---|---|---|
| the event_x columns | 40 | what each past filing was (Part 6) |
| `log_trading_days_ago` | 1 | how long ago |
| `abnormal_return`, `volume_change`, `car1`, `car5` | 4 | its reaction (if knowable yet) |
| `reaction=…` | 5 | its reaction class (if knowable yet) |
| `day0_known`, `volume_known`, `car1_known`, `car5_known` | 4 | 1 = shown, 0 = hidden |
| **total** | **54** | |

### Lines 37–45: `load_tables`, putting the data on the GPU once

```python
TABLE_KEYS = ["event_x", "company_x", "abnormal_return", "volume_change", "car1", "car5",
              "volume_present", "reaction_onehot", "trading_day", "company_start",
              "target_car1", "target_class"]


def load_tables(d, device):
    return {k: torch.as_tensor(d[k]).to(device) for k in TABLE_KEYS}
```

- `TABLE_KEYS`: the names of every per-event array a batch needs (Part 6 made
  them).
- The dict comprehension (Part 1) builds a new dict with the same names. For each
  one, `torch.as_tensor(...)` turns the NumPy array into a tensor (without copying)
  and `.to(device)` copies it onto the GPU.
- That's about 90 MB, copied **once** at the start. From then on, building a batch
  never has to move data from the CPU, and the GPU does the work.

`t` (for "tables") is the name this dict gets everywhere else.

### Lines 48–93: `history_batch`, the heart of the leakage rule

This one function decides **everything the model sees about the past**, for a
**whole batch** of events at once. It works on grids (one row per event in the
batch, one column per history slot) instead of looping over events. That's what
lets the GPU do it in a few operations.

`t` is the tables dict; `rows` is a tensor of the B events' row numbers.

To make it concrete, we'll follow a **tiny batch of two events**:

- event in row **7**, whose company's events start at row **4**, so it has 3 past
  events (rows 4, 5, 6);
- event in row **12**, whose company starts at row **11**, so it has 1 past event
  (row 11).

#### Line 61: how many past events each one gets

```python
history_len = torch.clamp(rows - t["company_start"][rows], max=cfg.MAX_HISTORY_LEN)
```

- `t["company_start"][rows]`: for each event, the row where its company starts →
  `[4, 11]`.
- `rows - ...`: how many rows lie between the company's start and the event →
  `[7-4, 12-11]` = `[3, 1]`. Because the data is sorted by company then date
  (Part 6, line 53), those rows are exactly its earlier events.
- `torch.clamp(..., max=128)`: cap at 128, the history limit.

Result: `history_len = [3, 1]`.

#### Lines 62–64: the grid and which slots are real

```python
L = int(history_len.max())
step = torch.arange(L, device=rows.device)
real = step[None, :] < history_len[:, None]
```

- `L`: the longest history in this batch (3). Every event gets L slots. `int(...)`
  brings that one number back from the GPU, because Python needs a plain number to
  size the grid.
- `step = [0, 1, 2]`: the slot numbers.
- `real`: a (B, L) True/False grid, True where a slot holds a real past event.
  Comparing a row `[0, 1, 2]` with a column `[3, 1]` broadcasts (Part 2):

```
            slot:   0      1      2
event row 7:       True   True   True      (3 real)
event row 12:      True   False  False     (1 real, then 2 padding slots)
```

#### Line 67: which row goes in each slot

```python
past = torch.where(real, rows[:, None] - history_len[:, None] + step[None, :], rows[:, None])
```

- `rows[:, None] - history_len[:, None] + step[None, :]`: the first past row is
  `row - history_len`, and the slots count up from there:
  - row 7: `7 - 3 + [0, 1, 2]` = `[4, 5, 6]`: its 3 past events, oldest first.
  - row 12: `12 - 1 + [0, 1, 2]` = `[11, 12, 13]`. Only slot 0 (row 11) is real;
    12 and 13 are not its past.
- `torch.where(real, A, B)`: element by element, take A where `real` is True, and B
  where it's False. Padding slots get the event's own row number, **just so every
  slot holds a valid row number** for the lookups below. Their contents are wiped
  to zero at the very end (line 93), so what they point at never matters.

Result: `past = [[4, 5, 6], [11, 12, 12]]`.

#### Line 69: how many trading days ago

```python
gap = t["trading_day"][rows][:, None] - t["trading_day"][past]
```

**Indexing with a grid** is the key trick in this function.
`t["trading_day"][past]`, where `past` is a (2, 3) grid of row numbers, returns a
(2, 3) grid in which each row number has been replaced by that row's trading day.
In one operation it looks up every slot of every event.

`t["trading_day"][rows][:, None]` is each event's own day 0 as a column, and the
subtraction broadcasts, so `gap` is (B, L): trading days ago for every slot.

#### Lines 70–73: what is knowable yet

```python
day0_known = (gap >= cfg.MIN_GAP_DAY0) & real
car1_known = (gap >= cfg.MIN_GAP_CAR1) & real
car5_known = (gap >= cfg.MIN_GAP_CAR5) & real
volume_known = day0_known & t["volume_present"][past]
```

The leakage rule from Part 5, applied to the whole grid at once. Each is a (B, L)
True/False grid. `& real` makes sure padding slots are never marked "known".

#### Lines 76–80: the time columns

```python
gap = gap.double()
time_cols = [torch.log1p(gap)]
if cfg.USE_TIME_CYCLES:
    for p in cfg.TIME_CYCLE_PERIODS:
        time_cols += [torch.sin(2 * torch.pi * gap / p), torch.cos(2 * torch.pi * gap / p)]
```

- `gap.double()`: convert to 64-bit decimals for these calculations. With gaps
  around 2,700 days, `2π × gap / p` reaches about 3,400, and 32-bit numbers would
  lose precision in sin/cos (an error of up to 0.0003, measured). It's converted
  back to 32-bit when the columns are built.
- `log(1 + gap)`, plus, if switched on, the sin/cos "clock positions" (Part 5).

#### Lines 82–93: build the table and wipe the padding

```python
column = lambda x: x.float()[..., None]
history = torch.cat([
    t["event_x"][past],
    *[column(c) for c in time_cols],
    column(t["abnormal_return"][past] * day0_known),
    column(t["volume_change"][past] * volume_known),
    column(t["car1"][past] * car1_known),
    column(t["car5"][past] * car5_known),
    t["reaction_onehot"][past] * car1_known[..., None],
    column(day0_known), column(volume_known), column(car1_known), column(car5_known),
], dim=-1)
return history * real[..., None], history_len
```

- `column`: a tiny helper. It turns a (B, L) grid into a (B, L, 1) grid of 32-bit
  numbers, so it can be glued on as one more column. `...` means "all the existing
  dimensions", and `None` adds a new last one.
- `t["event_x"][past]`: grid indexing again. `event_x` is (events, 40), so this
  gives **(B, L, 40)**: the 40 "what was filed" numbers for every slot.
- `t["abnormal_return"][past] * day0_known`: **the hiding**. It's the same trick as
  before (Part 2): multiplying by False gives 0. One multiplication applies the
  rule to every slot of every event.
- The reaction one-hot is (B, L, 5), so the mask gets an extra axis
  (`[..., None]`) to cover all 5 columns.
- `torch.cat([...], dim=-1)`: glue everything along the last dimension, giving
  **(B, L, 54)**.
- `history * real[..., None]`: multiply every padding slot by False, which makes
  it **all zeros**. That includes the event's own data that line 67 put there as a
  placeholder.
- It returns the table and `history_len` (the model needs to know where the
  padding starts).

**Is this the same as building one event at a time?** Yes. It was checked against
the previous one-event-at-a-time version on 5,793 events, including 500 same-day
cases and 300 first events. The batches are bit-identical, and padding is exactly
zero. A full scan of all 411,685 events found **zero** reaction values visible
before their window closed.

### Lines 96–99: `history_for`, one event's history for inspecting

```python
def history_for(d, i):
    history, history_len = history_batch(load_tables(d, "cpu"), torch.tensor([i]))
    return history[0, :history_len[0]].numpy()
```

A convenience wrapper so you can look at one event (Part 10):

- `load_tables(d, "cpu")`: the tables, kept on the CPU (nothing is copied).
- `torch.tensor([i])`: a "batch" of one event.
- `history[0, :history_len[0]]`: the only event in the batch, without padding.
- `.numpy()`: back to a NumPy table.

Because it calls `history_batch`, you see **exactly** what training sees. There
is only one copy of the leakage rule.

### Lines 102–111: `batches`, which serves the batches

```python
def batches(t, rows, shuffle):
    if shuffle:
        rows = rows[torch.randperm(len(rows), device=rows.device)]
    for start in range(0, len(rows), cfg.BATCH_SIZE):
        r = rows[start:start + cfg.BATCH_SIZE]
        history, history_len = history_batch(t, r)
        current = torch.cat([t["event_x"][r], t["company_x"][r]], dim=-1)
        yield history, history_len, current, t["target_car1"][r], t["target_class"][r]
```

This replaces a DataLoader, and it's a generator (Part 1): each time the training
loop asks for a batch, it builds one.

- `torch.randperm(n)`: the numbers 0 … n-1 in random order. Using it to index
  `rows` shuffles them. It's done on the GPU, and `torch.manual_seed` makes it
  repeatable.
- `range(0, len(rows), cfg.BATCH_SIZE)` counts 0, 64, 128, …, the start of each
  batch. `rows[start:start + 64]` is that batch's 64 row numbers (the last batch
  may be smaller).
- `history_batch(t, r)`: the batch's histories.
- `current`: each event's 40 event_x numbers followed by its 23 company_x numbers,
  giving **(B, 63)**. Nothing from the event's own reaction.
- `yield` hands over the 5 tensors: history (B, L, 54), history_len (B,), current
  (B, 63), the CAR targets (B,) and the class targets (B,). They're **already on
  the GPU**.

### Lines 114–124: `get_device`

```python
def get_device():
    if torch.cuda.is_available():
        try:
            (torch.zeros(8, device="cuda") + 1).sum().item()
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        except RuntimeError as e:
            print(f"GPU found but unusable with this PyTorch build ({str(e).splitlines()[0]}).\n"
                  "Using CPU. See the install note at the top of this file.")
    return torch.device("cpu")
```

- `torch.cuda.is_available()`: is there an NVIDIA GPU PyTorch can see? This can
  say True even when this PyTorch version can't actually run code on that GPU.
  That's exactly what happened with your RTX 5060 Ti and PyTorch 2.5.
- So it **tests for real**: make 8 zeros on the GPU, add 1, sum them, and pull
  the answer back with `.item()`. If that works, the GPU works.
- `try` / `except RuntimeError as e`: if the test fails, print why (only the first
  line of the error, via `.splitlines()[0]`) instead of crashing.
- If the GPU worked, the function has already returned `"cuda"`. Otherwise it
  falls through to the last line and returns the CPU.

### Lines 127–146: `run_epoch`, one pass over the data

```python
def run_epoch(model, t, rows, pos_weight, optimizer=None):
    training = optimizer is not None
    model.train(training)
    car_loss = nn.SmoothL1Loss() if cfg.TASK == "regression" else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    class_loss = nn.CrossEntropyLoss()
    total = 0.0
    with torch.set_grad_enabled(training):
        for history, lengths, current, car, cls in batches(t, rows, shuffle=training):
            car_pred, logits = model(history, lengths, current)
            target = car if cfg.TASK == "regression" else (car > 0).float()
            loss = car_loss(car_pred, target) + cfg.REACTION_LOSS_WEIGHT * class_loss(logits, cls)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(car)
    return total / len(rows)
```

The same function serves both training and validation; the only difference is
whether you hand it an optimizer.

- `training = optimizer is not None`: True if an optimizer was given (training),
  False for validation. `optimizer=None` in the signature means the argument is
  optional.
- `model.train(training)`: dropout on for training, off for validation.
- `car_loss = ... if ... else ...`: an inline if (Part 1). Huber loss for
  regression; `BCEWithLogitsLoss(pos_weight=pos_weight)` for direction. `pos_weight`
  multiplies the loss on every **up** event. It's computed once in `main`
  (Lines 185–188) and passed in, so training and validation use the same
  weighting. `class_loss` is the 5-class
  cross-entropy either way (Part 3).
- `target = ...`: what the CAR output is compared against. For regression, the CAR
  itself. For direction, `(car > 0).float()`: `car > 0` gives True/False per event,
  and `.float()` turns that into 1.0 (up) or 0.0 (down). `BCEWithLogitsLoss` then
  pushes the logit above 0 for up events and below 0 for down events.
- `total = 0.0`: a running total of the loss.
- `with torch.set_grad_enabled(training):` record gradients only when training.
- `for ... in batches(t, rows, shuffle=training):` `batches` hands out one batch
  at a time (5 tensors), unpacked into 5 names. Training batches are shuffled;
  validation batches aren't. The tensors are already on the GPU, so there's no
  `.to(device)` step.
- `car_pred, logits = model(...)`: the **forward pass** (Part 7).
- `loss = ...`: one number: CAR loss + 0.3 × class loss.
- When training, the three steps from Part 3:
  - `optimizer.zero_grad()`: clear old gradients. PyTorch **adds** new gradients
    to old ones by default, so without this, batches would pile up.
  - `loss.backward()`: compute the gradient for all 83,270 weights.
  - `optimizer.step()`: nudge every weight.
- `total += loss.item() * len(car)`: `loss` is the **average** over the batch, so
  multiply by the batch size to get a total. The last batch may be smaller than
  64, and this keeps the overall average exact.
- `return total / len(rows)`: the average loss per example over the whole
  epoch.

### Lines 149–157: `predict`

```python
@torch.no_grad()
def predict(model, t, rows):
    model.eval()
    car_pred, class_pred = [], []
    for history, lengths, current, _, _ in batches(t, rows, shuffle=False):
        c, logits = model(history, lengths, current)
        car_pred.append(c.cpu().numpy())
        class_pred.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(car_pred), np.concatenate(class_pred)
```

- `@torch.no_grad()`: no gradient recording; we're only predicting.
- `model.eval()`: dropout off.
- `for ..., _, _ in batches(...)`: the targets aren't needed here, so they go
  into `_`. Batches come in a fixed order (no shuffle), matching `test_rows`.
- `c.cpu().numpy()`: bring the predictions back from the GPU and turn them into
  NumPy.
- `logits.argmax(-1)`: the predicted class is the one with the highest score
  (along the last dimension, the 5 classes).
- `np.concatenate(...)`: join the per-batch arrays into one long array.

### Lines 160–168: start of `main`

```python
assert cfg.TASK in ("regression", "direction"), f"unknown TASK {cfg.TASK!r}"
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
with open(cfg.EVENTS_PKL, "rb") as f:
    d = pickle.load(f)
print(f"Settings: USE_TIME_CYCLES=..., DROP_LOOKAHEAD_FEATURES=..., CAUSAL_READOUT=..., TASK=..., SEED=...")
```

- The `assert` stops immediately if `TASK` is misspelled. Without it, a typo like
  `"regresion"` would quietly run direction mode, because the code checks
  "regression or not". `{cfg.TASK!r}` prints the value with quotes so the typo is
  easy to spot.
- Fix both random-number generators so the run is repeatable (Part 5).
- Load the 141 MB data dict from the prepare step.
- Print the settings, so every results log records which version of the
  experiment produced it.

### Lines 170–174: the time-based split

```python
dates = d["event_date"]
train_rows = np.flatnonzero(dates < np.datetime64(cfg.TRAIN_END))
val_rows = np.flatnonzero((dates >= np.datetime64(cfg.TRAIN_END)) & (dates < np.datetime64(cfg.VAL_END)))
test_rows = np.flatnonzero(dates >= np.datetime64(cfg.VAL_END))
```

- `dates < np.datetime64("2025-01-01")`: a True/False array, one per event.
- `np.flatnonzero(...)`: the row numbers where it's True.
- Validation uses `&` to combine two conditions: on or after TRAIN_END **and**
  before VAL_END.

Your data splits as **327,754 train**, then validation (first half of 2025), then
test (July 2025 onwards).

**One subtle point:** a test event's *history* can contain training-period events.
That's correct and intended. In real life you'd know a company's past when
predicting its future. What matters is that the *events being scored* in the test
are later than anything the model **trained on**.

### Lines 176–178: the device, and the data onto it

```python
device = get_device()
t = load_tables(d, device)
on_device = lambda rows: torch.as_tensor(rows, device=device)
```

- `get_device()`: the GPU if it works, otherwise the CPU.
- `load_tables(d, device)`: copy all the event data onto that device, once.
- `on_device`: a tiny function (Part 1) that turns a NumPy array of row numbers
  into a tensor on the device. It's needed because indexing GPU tables needs GPU
  row numbers. The NumPy versions (`train_rows` etc.) are kept for the final scoring.

### Lines 179–183: build the model and optimizer

```python
n_history = len(history_columns(d))
n_current = len(d["event_columns"]) + len(d["company_columns"])
model = Tier0Model(n_history, n_current, n_classes=len(d["reaction_classes"]),
                   causal_readout=cfg.CAUSAL_READOUT).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
```

- The input sizes are **counted from the data** (54 and 63), not typed in by hand.
  Add a feature in the prepare step and the model resizes itself.
- `causal_readout=cfg.CAUSAL_READOUT`: which summary design to build (Part 7).
- `.to(device)`: move all the model's weights to the GPU.
- `torch.optim.AdamW(model.parameters(), lr=...)`: give the optimizer the list of
  every learnable weight, and the step size.

### Lines 185–188: the class weight for the direction task

```python
train_car = d["target_car1"][train_rows]
pos_weight = torch.tensor((train_car <= 0).sum() / (train_car > 0).sum(), device=device)
```

- `train_car`: the CAR of every **training** event (it's also used for the
  baselines later).
- `(train_car <= 0).sum()`: how many training events went down (True counts as
  1). `(train_car > 0).sum()`: how many went up. `pos_weight` = down ÷ up.
- `torch.tensor(..., device=device)`: the loss runs on the GPU, so the weight must
  live there too.

**What it does.** If up events were rarer, a model could score well by mostly
saying "down". `pos_weight` makes each up event count **more** in the loss by
exactly the imbalance, so getting the rare side wrong costs as much in total as
getting the common side wrong.

**Rules it follows.** It's counted on the **training split only**; using
validation or test labels would leak information from the future. And it's
computed **once, before training**, so it never changes between epochs or
batches.

**In your data it barely matters.** The training split has 161,991 up and 165,763
down events, so `pos_weight` = **1.023**. The weighting is there for correctness,
but don't expect it to change results much. If a trained model still collapses to
one side, imbalance isn't the reason.

### Lines 190–193: the training loop

```python
for epoch in range(1, cfg.EPOCHS + 1):
    train_loss = run_epoch(model, t, on_device(train_rows), pos_weight, optimizer)
    val_loss = run_epoch(model, t, on_device(val_rows), pos_weight)
    print(f"epoch {epoch:2d}/{cfg.EPOCHS}  train loss {train_loss:.4f}  validation loss {val_loss:.4f}")
```

- `range(1, 16)` counts 1 to 15. (A range's end is excluded, hence `+ 1`.)
- Each epoch trains on all training batches (optimizer given), then measures the
  validation loss (no optimizer, so nothing changes).

**How to read the printout:** both losses should fall at first. If the training
loss keeps falling while the validation loss starts **rising**, the model is
memorising the training data (overfitting).

### Lines 196–200: test predictions and the baselines

```python
car_pred, class_pred = predict(model, t, on_device(test_rows))
car_true, class_true = d["target_car1"][test_rows], d["target_class"][test_rows]
baseline = d["target_car1"][train_rows].mean()
majority = np.bincount(d["target_class"][train_rows]).argmax()
majority_direction = 1 if (train_car > 0).sum() >= (train_car < 0).sum() else -1
```

- Predict on the test period; get the true answers for the same rows (same order,
  because test batches aren't shuffled).
- **`baseline`**: the average CAR across the training set. A "model" that always
  predicts this number knows nothing. It's the bar to beat.
- **`majority`**: `np.bincount` counts how often each class number (0–4) appears
  in training, and `.argmax()` picks the most common. Always guessing that class
  is the classification bar to beat.
- **`majority_direction`**: was "up" or "down" more common in training? (`train_car`
  was defined at Lines 185–188.) `(train_car > 0).sum()` counts the positive CARs (True counts as 1), and likewise the
  negatives. It's +1 if positives are at least as common, otherwise -1. Always
  guessing that direction is the up/down bar to beat.

  Why not just use the sign of `baseline`, the average? Because a few huge moves
  can drag the average one way while most events go the other. In your training
  data the average is **positive** (+0.58%), but **more events are negative**
  (50.6% vs 49.4%). Guessing "up" scores only 0.465 on the test period, while
  guessing the commoner "down" scores 0.535. Using the average's sign would have
  made the bar 7 points too easy.

### Lines 204–205: R² against the baseline

```python
def r2_vs_baseline(mask):
    return 1 - ((car_true[mask] - car_pred[mask]) ** 2).sum() / ((car_true[mask] - baseline) ** 2).sum()
```

```
R² vs baseline = 1 - (model's squared errors) / (baseline's squared errors)
```

- **0** = the model is exactly as good as guessing the average.
- **above 0** = better. 0.02 means it removed 2% of the squared error.
- **below 0** = worse than guessing the average.

In return prediction, even a small positive number, measured honestly on later
data, is a real result. `mask` picks which test events to score.

### Lines 202–230: the results

The printout depends on `TASK`: `if cfg.TASK == "regression":` prints the
regression results, and `else:` prints the direction results. The reaction
accuracy line comes after both, so it's always printed.

**Regression results.**

```python
everything = np.ones(len(car_true), bool)
ordinary = np.abs(car_true) <= 0.5
```

- `everything`: a mask that's all True, meaning every test event.
- `ordinary`: only events whose CAR was within ±50%. Squared error is dominated by
  a handful of enormous moves (some are data glitches), which can make the
  all-events R² noisy. Reporting both shows whether the model works on normal
  events.

The prints show:

- **MAE** (mean absolute error): the average size of the miss, for the model and
  for the baseline. Lower is better.
- **R² vs baseline**: for all events, and for ordinary ones.
- **reaction accuracy**: the fraction of classes guessed right, model vs
  always-the-commonest-class. `class_pred == class_true` is True/False per event,
  and `np.mean` of True/False is the fraction that's True.
- **direction accuracy**: ignores size and only asks "did the prediction land on the
  same side of zero as the truth?" `np.sign` turns each number into -1, 0 or +1,
  and comparing signs gives True/False per event. The baseline always guesses
  `majority_direction`, the direction that was commoner in training (here
  "down"). Its score is just the share of test events on that side, about 53.5%
  here, which makes this a **harder** bar to clear than the class baseline.

**Direction results** (`TASK = "direction"`).

```python
up_true, up_pred = car_true > 0, car_pred > 0
print(... np.mean(up_pred == up_true) ... np.mean((majority_direction > 0) == up_true) ...)
recall_up, recall_down = np.mean(up_pred[up_true]), np.mean(~up_pred[~up_true])
print(f"  recall               up events {recall_up:.3f}   down events {recall_down:.3f}")
print(f"  balanced accuracy    model {(recall_up + recall_down) / 2:.3f}   baseline 0.500")
```

- `up_true`: True where the event really went up. `up_pred`: True where the
  model's logit is above 0, meaning it predicted up (above 0 is the same as a
  probability above 50%).
- **direction accuracy**: the fraction where `up_pred` matches `up_true`. The
  baseline always guesses the commoner training direction:
  `majority_direction > 0` is a single True ("always up") or False ("always
  down"), compared against every event.
- **recall, up events**: `up_pred[up_true]` keeps only the events that really
  went up (a True/False mask used as an index, Part 2) and asks what fraction the
  model called up.
- **recall, down events**: `~up_pred[~up_true]` does the same for the events that
  really went down. `~` means "not", so this is the fraction the model called down.
- **balanced accuracy**: the average of the two recalls, so up and down events
  count **equally** however many of each there are. Its baseline is always
  **0.500**: always guessing one side gives recalls of 1 and 0, and (1 + 0) / 2 =
  0.5. Unlike plain accuracy, it can't be inflated by favouring the commoner side.

**Why recall matters.** Accuracy alone can hide a lazy model. A model that always
says "down" scores the same accuracy as the baseline (about 53.5%) while learning
nothing; recall exposes it as **0.000 on up events and 1.000 on down events**. (An
untrained model produced exactly that pattern in testing.) A model that has
learned something shows **both** recalls well above 0, and accuracy above the
baseline.

- Format codes: `:.5f` = 5 decimals; `:+.4f` = 4 decimals with a sign;
  `:.1%` = a percentage.

### Lines 233–234

```python
if __name__ == "__main__":
    main()
```

Train only when this file is run directly. That's why you can
`import tier0_train` and call `history_for` yourself (Part 10) without starting a
training run.

---

## Part 9: One real event's full journey

Let's follow one real event from your data through every step: **JPMorgan
(JPM), event P2E200266, an 8-K dated 2016-01-25.** It's an "Other" filing (item
8.01), filed on time. Its real 3-day reaction was **CAR1 = +0.014840 (+1.48%)**,
class **Positive**. That's the answer the model will be scored against.

### Step 1: in the prepared data (tier0_prepare_data.py)

After sorting, this event is **row 200,265**. JPMorgan's events occupy rows
200,260 to 200,587. Its day 0 is trading day **14** of the calendar (the 15th
trading day of 2016).

Its **current** vector (63 numbers) is mostly zeros, because one-hot columns are
mostly zero. The non-zero ones:

| column | value | meaning |
|---|---|---|
| `event_category=Other` | 1 | it's an "Other" filing |
| `filing_timeliness=On-time` | 1 | filed on time |
| `item=8.01` | 1 | lists item 8.01 |
| `num_items_disclosed` | 1 | one item |
| `sector=Financial Services` | 1 | |
| `size_category=Mega-cap` | 1 | |
| `log_market_cap` | 27.504 | log(1 + $880bn-ish) |
| `log_revenue_ttm` | 25.951 | |
| `log_employees` | 12.678 | log(1 + ~320,000) |
| `volatility_tier=Low` | 1 | |
| `log_trailing_volatility_1y` | 0.203 | log(1 + 0.225), i.e. about 22.5% yearly volatility |

`log_days_since_prev_filing` is 0: JPMorgan filed another 8-K **the same day**.
Remember that; it's about to matter.

### Step 2: building its history (history_batch)

The 5 earlier JPMorgan events are rows 200,260–200,264. Their trading days are
0, 8, 8, 10 and 14; the current event's is 14. So:

```
gap = 14 - [0, 8, 8, 10, 14] = [14, 6, 6, 4, 0]
```

Now the leakage rule decides what's visible (real output of the history builder):

| past event | date | type | gap | AR shown? (≥2) | CAR1 shown? (≥3) | CAR5 shown? (≥7) |
|---|---|---|---|---|---|---|
| P2E200261 | 2016-01-04 | Other | 14 | ✅ -0.0146 | ✅ -0.0139 | ✅ -0.0450 |
| P2E200262 | 2016-01-14 | Earnings | 6 | ✅ -0.0017 | ✅ -0.0025 | ❌ hidden (0) |
| P2E200263 | 2016-01-14 | Earnings | 6 | ✅ -0.0017 | ✅ -0.0025 | ❌ hidden (0) |
| P2E200264 | 2016-01-19 | Governance | 4 | ✅ -0.0011 | ✅ -0.0139 | ❌ hidden (0) |
| P2E200265 | 2016-01-25 | Other | **0** | ❌ hidden | ❌ hidden | ❌ hidden |

(The "shown" values are after `signed_log`, so a raw -0.0147 becomes -0.0146.
Rows 2 and 3 are two filings JPMorgan made on the same day, so they share one
reaction.)

**Look at the last row.** P2E200265 was filed the **same day** as our event, so
they share one price move. Its raw CAR1 is **+0.0148**, the very answer we're
trying to predict (+0.014840). Without the leakage rule, the model could simply
copy it and look like a genius. With the rule, gap 0 fails every threshold, so all
its reaction values are 0 and all its `_known` flags are 0. The model still sees
that a same-day "Other" filing exists (its event_x columns), which is legitimate,
public information, but not how the market reacted.

Similarly for rows 1–3: their 7-day CAR (CAR[-1,+5]) runs into our event's
window, so it's hidden. Their 3-day CAR finished in time, so it's shown.

The result is a history table of shape **(5, 54)**: 5 past events, 54 numbers each.

### Step 3: into a batch (batches)

`batches` picks this event's row number (200,265) as one of 64 in a batch.
`history_batch` builds all 64 histories at once on the GPU: this event gets 5 real
slots, followed by zero-filled padding slots up to the longest history in the
batch. Its `history_len` is 5, its current vector is the 63 numbers above, and its
targets are 0.01484 and class 3 ("Positive" in `REACTION_CLASSES`).

### Step 4: through the model (shapes for this one example)

```
history            (1, 5, 54)
history_proj   →   (1, 5, 64)       each past event becomes 64 numbers
+ summary token →  (1, 6, 64)       position 0 = summary, positions 1–5 = the events
is_padding         [F, F, F, F, F, F]   (no padding when it's alone)
encoder (2 layers) (1, 6, 64)       the summary attends to the 5 events
[:, 0]         →   (1, 64)          the summary: "JPMorgan's recent history"

current            (1, 63)
current_proj   →   (1, 64)

cat            →   (1, 128)
combine        →   (1, 64)
car_head       →   (1,)             predicted CAR1
class_head     →   (1, 5)           5 class scores
```

Inside the encoder, the summary token's attention decides how much to draw from
each of the 5 events. A trained model might learn that the old 2016-01-04 event
(the only one with everything known) or the earnings pair matters most here. You
can't see the weights in this code, but they're learned exactly as in Part 4's
example.

### Step 5: the loss and learning (run_epoch)

Before training, the weights are random and so is the output: an untrained model
predicted **-0.8425** (-84%!) for this event. Huber loss with error
-0.8425 - 0.0148 = -0.857 is 0.5 × 0.857² ≈ 0.367; the class loss adds to it.
`loss.backward()` works out how each of the 83,270 weights contributed, and
`optimizer.step()` nudges them so the prediction moves towards +0.0148. Over
tens of thousands of such nudges (5,122 batches × 15 epochs = 76,830), the weights settle on
whatever patterns genuinely help across all events.

### Step 6: scoring (only if it's a test event)

This event is from 2016, so it's in the **training** set and is never scored. A
July 2025+ event would go through exactly the same steps, but the model would only
predict it (`predict`), never learn from it, and its error would count towards the
final MAE and R² against the baseline.

---

## Part 10: Running it and experimenting

### Which Python to use

Use the **project `.venv`**: it has PyTorch 2.11 built for CUDA 12.8, which runs
on your RTX 5060 Ti. The system Python's older PyTorch can't use the GPU.

```powershell
cd C:\Users\Natha\Desktop\DataMining\phase2\network
..\..\.venv\Scripts\python.exe tier0_prepare_data.py     # about a minute; only when the data changes
..\..\.venv\Scripts\python.exe tier0_train.py            # about 17 min on the GPU (15 epochs)
```

Or in VS Code, choose the `.venv` interpreter (Ctrl+Shift+P → "Python: Select
Interpreter") and press Run. You should see `Using GPU: NVIDIA GeForce RTX 5060 Ti`.

### Look at one example yourself

Run this in a Python shell started in this folder (`..\..\.venv\Scripts\python.exe`):

```python
import pickle, numpy as np, pandas as pd
import tier0_config as cfg, tier0_train as tr      # importing doesn't start training

d = pickle.load(open(cfg.EVENTS_PKL, "rb"))
i = int(np.flatnonzero(d["event_id"] == "P2E200266")[0])     # any event_id you like
history = pd.DataFrame(tr.history_for(d, i), columns=tr.history_columns(d))
print(history[["log_trading_days_ago", "car1", "car5", "car1_known", "car5_known"]])
```

`history_for` gives the table, `history_columns` gives its column names, and the
DataFrame makes it readable. For this event it prints:

```
   log_trading_days_ago      car1      car5  car1_known  car5_known
0              2.708050 -0.013915 -0.044986         1.0         1.0
1              1.945910 -0.002538 -0.000000         1.0         0.0
2              1.945910 -0.002538 -0.000000         1.0         0.0
3              1.609438 -0.013910 -0.000000         1.0         0.0
4              0.000000  0.000000  0.000000         0.0         0.0
```

`2.708` is log(1 + 14), so 14 trading days ago; `1.946` = log(1 + 6); `1.609` =
log(1 + 4). The same-day row is all zeros. **`-0.000000` is just zero:** a
negative value multiplied by False becomes "negative zero", which computers print
with a minus sign but which equals 0 exactly.

### Experiments you can run by editing tier0_config.py

| setting | try | question it answers |
|---|---|---|
| `USE_TIME_CYCLES` | `True` | Does cyclical timing (weekly/quarterly) help? |
| `DROP_LOOKAHEAD_FEATURES` | `True` (then re-run **prepare**) | Are results leaning on the 2026 snapshot? |
| `CAUSAL_READOUT` | `True` | Summary token over everything, or read the last event with a causal mask? |
| `TASK` | `"direction"` | Is up/down easier to predict than the size of the move? |
| `MAX_HISTORY_LEN` | 32, 64, 256 | How much history matters? |
| `SEED` | 0, 1, 2 | How much is noise? Run every experiment with several seeds |

`DROP_LOOKAHEAD_FEATURES` changes what the prepare step saves, so run
`tier0_prepare_data.py` again after changing it. The others only affect training.

**The fair-comparison rule:** change **one** setting at a time, run each version
with seeds 0, 1 and 2, and compare the test numbers. If the gap between versions
is smaller than the spread between seeds, you haven't found an effect.

### Reading the results

```
TEST RESULTS (model vs. naive baseline)
  CAR MAE              model 0.0xxxx   baseline 0.0xxxx
  R^2 vs baseline      all events +0.00xx   |CAR| <= 50% (..% of events) +0.00xx
  reaction accuracy    model 0.xxx   baseline 0.xxx
  direction accuracy   model 0.xxx   baseline 0.xxx
```

With `TASK = "direction"`, the CAR MAE and R² lines are replaced by direction
accuracy, `recall  up events 0.xxx  down events 0.xxx` and
`balanced accuracy  model 0.xxx  baseline 0.500`. Above 0.500 balanced accuracy
means the model tells up from down better than chance.

- Model MAE **lower** than baseline MAE: it misses by less than "always guess the
  average" does.
- R² vs baseline **above 0**: it explains some of the variation. Watch the
  ordinary-events number especially, because the all-events number is swung by
  extreme moves.
- Reaction accuracy **above** the baseline: it beats "always guess the commonest
  class".
- Direction accuracy **above** the baseline: it calls up/down better than "always
  guess the commoner direction". Even a few points above ~53.5% is meaningful here.
- All at or below baseline: an honest, reportable result. This feature set
  doesn't predict reactions better than chance.

---

## Part 11: Glossary

| term | meaning |
|---|---|
| **8-K** | A form US companies file with the SEC to announce significant events |
| **abnormal return (AR)** | The stock's return minus the market's return that day |
| **CAR1 / CAR[-1,+1]** | Abnormal returns added up over trading days -1, 0, +1 around the filing; the model's target |
| **CAR5 / CAR[-1,+5]** | The same over days -1 to +5 |
| **day 0 / t0** | The first trading day on or after the event's report date |
| **trading day** | A day the stock market is open (no weekends or holidays) |
| **leakage** | Accidentally giving the model information it couldn't have known at prediction time |
| **feature** | One input number (e.g. `log_market_cap`) |
| **one-hot** | A category as 0/1 columns, one per value, exactly one of them 1 |
| **multi-hot** | Like one-hot, but several columns can be 1 (item codes) |
| **signed log** | sign(x) × log(1 + \|x\|); tames huge values, keeps the sign |
| **tensor** | PyTorch's array; can live on the GPU and track gradients |
| **shape** | The size of each dimension of an array, e.g. (64, 63, 54) |
| **batch (B)** | A group of examples processed together (64) |
| **padding** | Zero rows added so different-length histories fit one rectangle |
| **mask** | True/False marking positions to ignore |
| **d_model** | The width of the model's internal vectors (64) |
| **Linear layer** | Weighted sums of the inputs plus a constant; learnable |
| **weights / parameters** | The numbers training adjusts (83,270 here) |
| **activation (ReLU)** | max(0, x); makes the network able to learn curves, not just straight lines |
| **dropout** | Randomly zeroing 10% of values during training, against overfitting |
| **attention** | A learned weighted average: queries score keys, softmax makes weights, values are blended |
| **query / key / value** | What a position asks for / how it's labelled / what it passes on |
| **self-attention** | Every position attends to every position in the same sequence |
| **head** | One of several parallel attentions on a slice of the vector (4 heads × 16) |
| **softmax** | Turns scores into positive weights that add to 1 |
| **Transformer encoder layer** | Self-attention + feed-forward, each with residual + LayerNorm |
| **residual connection** | Adding a layer's input to its output, so it learns a change |
| **LayerNorm** | Rescales each vector to a standard range for stable training |
| **causal mask** | A mask that stops each position attending to later positions (only itself and earlier) |
| **summary token / [CLS]** | A learned extra position whose output is used as the sequence summary |
| **logits** | Raw class scores before softmax |
| **loss** | One number for how wrong the predictions are |
| **Huber / SmoothL1 loss** | Squared error for small misses, straight-line for big ones |
| **logit** | A raw score before squashing into a probability; for one yes/no question, above 0 means "yes" |
| **binary cross-entropy (BCEWithLogitsLoss)** | Cross-entropy for a single yes/no question, applied directly to a logit |
| **balanced accuracy** | The average of the per-side recalls; 0.5 = no better than guessing one side |
| **class weight (pos_weight)** | Makes each example of one class count more in the loss, to offset imbalance |
| **recall** | Of the events that really were X, the fraction the model called X |
| **cross-entropy** | -log(probability given to the true class) |
| **gradient** | For each weight, which way and how strongly it should change to reduce the loss |
| **backward pass** | `loss.backward()`: computing all the gradients |
| **optimizer (AdamW)** | The rule that uses gradients to update the weights |
| **learning rate** | The size of each update step (0.001) |
| **epoch** | One full pass over the training data |
| **overfitting** | Memorising the training data instead of learning general patterns |
| **train / validation / test** | Learn from / monitor during training / score once at the end |
| **walk-forward split** | Splitting by time so the model is always tested on later periods |
| **baseline** | A naive predictor (always the average) the model must beat |
| **MAE** | Mean absolute error: the average size of a miss |
| **R² vs baseline** | 1 - model squared error / baseline squared error; above 0 beats the baseline |
| **ablation** | Turning one ingredient off and on to measure what it contributes |
| **seed** | A starting value that makes "random" choices repeat exactly |
| **pickle (.pkl)** | Python's format for saving objects to a file |
| **device** | Where tensors live and are computed: the CPU or the GPU (`cuda`) |
