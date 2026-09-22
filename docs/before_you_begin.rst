Before you begin: Python, the terminal, and commands
====================================================

You do not need previous programming experience to follow the main Name That
Move tutorials. Most of the workflow uses ready-made commands: you enter a
command, press Enter, and read the response printed underneath it. Some later
sections also show Python code for people who want to build custom workflows.

The tools you will use
----------------------

**Python** is the programming language in which Name That Move is written.
Installing Python gives your computer the environment it needs to run the
package. You do not need to write a Python program before using the tutorial
commands. In simple terms, Python is the language used to build Name That Move
behind the scenes.

**The terminal** is a text-based application for communicating with your
computer. On macOS, open the application named **Terminal**. On Windows, open
**Windows Terminal** or **PowerShell**; the older **Command Prompt** can also
run many of the commands. Instead of clicking buttons, you type or paste a
short instruction and press Enter. For Name That Move, the terminal is the
window where you enter instructions and receive feedback from the tool.

**Name That Move commands** are short text instructions that you type or paste
into the terminal. Commands such as ``name-that-move-record``,
``name-that-move-train``, and ``name-that-move-live`` ask the Python code
behind Name That Move to perform particular functions. Words beginning with
``--`` are options that let you change what a command does.

Read a terminal example
-----------------------

A terminal usually displays a short prompt to show that it is ready and to
remind you which environment and folder you are working in. Prompts look
different on different computers. For example:

.. code-block:: text

   (your-environment) your-name@computer project-folder %

The terminal supplies this prompt; you do not need to change it. What you enter
after the prompt is a command. The examples in this documentation show only
the commands, so you can copy or paste them after the prompt and press Enter:

.. code-block:: console

   name-that-move-record --label my_move --session my_move_session_01

For easier reading, later examples place each ``--option`` on its own line:

.. code-block:: console

   name-that-move-record \
       --label my_move \
       --session my_move_session_01

The two versions do the same thing. On macOS and Linux, each ``\`` means that
the command continues on the next line. In Windows PowerShell, use the one-line
version without backslashes.

The program prints its status below the command. When the prompt returns, the
terminal is ready for the next command.

Useful terminal actions
-----------------------

* **Run a command:** press Enter.
* **Clear a command before running it:** press Control-C. On macOS and Linux,
  Control-U also clears the current line in many terminals.
* **Stop a running command:** press Control-C. The Name That Move recorder uses
  this as its normal stop signal and saves completed windows before exiting.
* **Repeat or edit a recent command:** press the Up Arrow key.
* **Paste:** use Command-V on macOS or Control-V in Windows Terminal and
  PowerShell.

Control-C does not close the terminal or damage the package. It either clears
the command you have not run yet or asks the current program to stop.

A preview of using Name That Move
---------------------------------

Here is a condensed preview of how you will use Name That Move. Each step and
unfamiliar term is explained in more detail as you move through the
documentation, so you do not need to understand everything yet:

1. create a dedicated Python environment;
2. install Name That Move into that environment;
3. run commands to record, train, and make predictions; and
4. read the terminal messages to find saved data, models, and results.

The environment keeps this project's Python tools separate from other
projects on your computer. When the tutorials say to *activate the
environment*, that means selecting this prepared workspace before running a
Name That Move command.

It is normal for unfamiliar terminal output to look dense at first. You do
not need to understand every printed line. The tutorials identify the messages
that matter for each step, and you are welcome to ask when an instruction or
message is unclear.

When you are ready, continue to :doc:`installation`.

.. container:: feedback-invitation

   **Questions and feedback are welcome**

   If a command or explanation is unclear, please `open a GitHub Issue
   <https://github.com/RoseZiyuXu/name-that-move/issues>`_. If GitHub is
   unfamiliar or your message involves private participant data or unpublished
   artistic material, please do not hesitate to `contact Rose
   <https://github.com/RoseZiyuXu>`_.
