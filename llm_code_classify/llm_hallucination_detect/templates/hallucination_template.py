"""
Prompt templates for hallucination detection in code.
"""

# Main hallucination detection prompt template
HALLUCINATION_DETECTION_PROMPT = """
I want you to act as a code judge. Given the function signature with description, and the generated code(The first definition is a docstring-only declaration,which isn't the gernerated code) your objective
is to detect if the generated code has defects, incorrect code or hallucinations. Carefully examine the code I provide, classify and label the issues using the given hallucination types in the format of "X.Y" (where X represents the major hallucination type number and Y represents the specific subtype number), and give a detailed description of the error, including the specific line number where the error occurs and a brief explanation of the cause.
Hallucination Types:
Data Type and Structure Issues
1.1 Data Type Mismatch: The model has a vague understanding of the data type of the object and parameter values. The generated code tries to perform operations with mismatched types or operations that break the rules. For example, if a function expects an integer as a parameter, but the code passes a string.
1.2 Misunderstanding of Data Structure: The model misinterprets the data structure of the operation object, causing the code to attempt to access non - existent array indices or dictionary keys.
Logical Issues
2.1 Overall Intention Conflict: The code implementation significantly deviates from the problem requirements, with major deviations in the overall algorithmic logic.
2.2 Context Inconsistency: The LLM struggles to interpret or maintain context during code generation (including input context and generated content), losing direction and failing to preserve strict contextual consistency. Examples include using undefined variables or misspelling variable names.
2.3 Local Intention Conflict: The general code implementation aligns with problem requirements, but specific details cause test failures.
2.4 Incomplete Code: The generated code lacks completeness, with no code related to input/output processing.
Issues of Duplicated and Ineffective Code
3.1 Code Duplication: Excessive repetition of certain code snippets in the generated content leads to unnecessary redundancy and inefficiency in the code. It can be divided into two sub - categories: duplication of the input context (copying and pasting existing code context snippets in the output) and duplication in the generated code (repeated code snippets within the generated code).
3.2 Dead Code: The generated code segment contains code that will not be executed or code whose execution results are not used in any other calculations.
External Dependency and Knowledge Issues
4.1 API Knowledge Conflict: The code contradicts the knowledge embedded in the API, shown as incorrect API calls, such as missing parameters or calling an unimported API.
4.2 External Source Conflict: When dealing with external knowledge sources, there are significant memory - related problems, causing the code to attempt to import non - existent modules or fail to load modules correctly.
Robustness and Security Issues
5.1 Robustness Issues: The code fails or throws an exception under specific boundary conditions, lacking the necessary exception handling.
5.2 Security Vulnerabilities: The code contains security vulnerabilities or memory leak problems.
Resource Issues
6.1 Exceeding Resource Constraints: The model underestimates the resource consumption in data processing, leading the code to fail due to exceeding physical limitations. For example, creating a very large list and continuously increasing its size in an infinite loop, resulting in a memory overflow.
Syntax Issues
7.1 Syntax Errors: The generated code has syntax errors and cannot be compiled.
7.2 Programming Language Confusion: The output contains natural language instead of code, or natural language is mixed in the code lines.

Required Output Format(you needn't output other information such as correct code)
Analysis Results
label: [X.X]
description: [Hallucination description with line number]
([] need to be contained in the output,not use Markdown formatting in your descriptions. Do not use asterisks, backticks or any formatting symbols.)
...

Code to Analyze:
{code}
""" 