## Self-Modifying Coding Agent

### 1. The Core Loop: Editing Code via Bash

An LLM cannot directly interact with your computer; it only outputs text. To let it run Bash commands, you wrap the LLM in a loop using **Tool Calling (Function Calling)**:

1. **Define the Tool:** You give the LLM a system prompt stating it has a tool called `run_bash(command)`.
2. **Intercept the Output:** When you ask the bot to fix a bug, the LLM outputs a structured request, such as `run_bash("grep -rn 'login' src/")`.
3. **Execute Locally:** Your Python script intercepts that string, runs it on your OS using Python’s `subprocess` module, and captures the terminal output (`stdout` and `stderr`).
4. **Feed It Back:** Your script sends the terminal output back to the LLM. The LLM reads the result and decides the next Bash command (like using `cat`, `sed`, or `python -c` to rewrite a file).

*Pro Tip:* Pure Bash editing (`sed` or `awk`) is notoriously error-prone for LLMs because of quote escaping. Most Cursor-like agents give the LLM a dedicated `edit_file(path, old_text, new_text)` tool alongside Bash, using Bash primarily for navigating, installing packages, and running tests.

---

### 2. How It Can Change Its Own Code

If your Python chatbot edits its own `.py` file while running, nothing happens immediately because the old code is already loaded in RAM. To make the changes take effect **without crashing the bot**, you use a two-part architecture:

* **The Supervisor / Worker Pattern**
You split your app into two scripts: a tiny, unchangeable `supervisor.py` and the main `agent.py`.
1. `supervisor.py` launches `agent.py` as a subprocess and manages the chat history.
2. You tell the agent, *"Add a new tool to your own code that lets you search the web."*
3. `agent.py` uses its Bash tool to edit `agent.py` on the hard drive and saves it.
4. Before restarting, `agent.py` runs `python -m py_compile agent.py` via Bash to verify it didn't write a syntax error.
5. `agent.py` exits with a specific status code (e.g., `exit(42)`), signaling `supervisor.py` to immediately restart `agent.py` with the newly saved code and pass the conversation history right back in.

---

### 3. Essential Safeguards (Preventing "Brain Surgery" Crashes)

If an agent makes a typo while editing its own core file and restarts, it will crash on boot—and since it is dead, it can no longer run Bash commands to fix its own mistake. To prevent this:

* **Automated Git Rollbacks:** Have the `supervisor.py` script automatically run `git commit` before the agent modifies itself. If the new `agent.py` crashes within 5 seconds of booting, the supervisor automatically runs `git reset --hard HEAD~1` to resurrect the last working version and tells the agent, *"Your last self-edit crashed the system; here is the traceback."*
* **Docker Sandboxing:** Because the agent has raw Bash access, a hallucinated `rm -rf /` will wipe your hard drive. Always run self-modifying Bash agents inside a Docker container or virtual environment.