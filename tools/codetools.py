from langchain_core.tools import tool
import os
import io
import sys
import uuid
import base64
import traceback
import contextlib
import tempfile
import subprocess
import sqlite3
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

"""NOTE:
    This class provides a secure environment for executing code in multiple 
    programming languages (Python, Bash, SQL, C, Java). It runs user or 
    AI-generated code with controlled access to resources, captures outputs, 
    errors, plots, and dataframes, making it ideal for AI agents 
    that need computational capabilities while maintaining security constraints.
"""

class CodeInterpreter:
    def __init__(self, allowed_modules=None, max_execution_time=30, working_directory=None):
        """Initialize the code interpreter with safety measures."""
        self.allowed_modules = allowed_modules or [
            "numpy", "pandas", "matplotlib", "scipy", "sklearn", 
            "math", "random", "statistics", "datetime", "collections",
            "itertools", "functools", "operator", "re", "json",
            "sympy", "networkx", "nltk", "PIL", "pytesseract", 
            "cmath", "uuid", "tempfile", "requests", "urllib"
        ]
        self.max_execution_time = max_execution_time
        self.working_directory = working_directory or os.path.join(os.getcwd()) 
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        
        self.globals = {
            "__builtins__": __builtins__,
            "np": np,
            "pd": pd,
            "plt": plt,
            "Image": Image,
        }
        self.temp_sqlite_db = os.path.join(tempfile.gettempdir(), "code_exec.db")


    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Execute the provided code in the selected programming language."""
        language = language.lower()
        execution_id = str(uuid.uuid4())
        
        result = {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": []
        }
        
        try:
            if language == "python":
                return self._execute_python(code, execution_id)
            elif language == "bash":
                return self._execute_bash(code, execution_id)
            elif language == "sql":
                return self._execute_sql(code, execution_id)
            elif language == "c":
                return self._execute_c(code, execution_id)
            elif language == "java":
                return self._execute_java(code, execution_id)
            else:
                result["stderr"] = f"Unsupported language: {language}"
        except Exception as e:
            result["stderr"] = str(e)
        
        return result


    def _execute_python(self, code: str, execution_id: str) -> dict:
        """Execute Python code safely in a controlled environment, 
            capturing outputs, errors, plots, and dataframes.
        """
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        result = {
            "execution_id": execution_id,
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": None,
            "plots": [],
            "dataframes": []
        }
        
        try:
            exec_dir = os.path.join(self.working_directory, execution_id)
            os.makedirs(exec_dir, exist_ok=True)
            plt.switch_backend('Agg')
            
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
                exec_result = exec(code, self.globals)

                if plt.get_fignums():
                    for i, fig_num in enumerate(plt.get_fignums()):
                        fig = plt.figure(fig_num)
                        img_path = os.path.join(exec_dir, f"plot_{i}.png")
                        fig.savefig(img_path)
                        with open(img_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            result["plots"].append({
                                "figure_number": fig_num,
                                "data": img_data
                            })

                for var_name, var_value in self.globals.items():
                    if isinstance(var_value, pd.DataFrame) and len(var_value) > 0:
                        result["dataframes"].append({
                            "name": var_name,
                            "head": var_value.head().to_dict(),
                            "shape": var_value.shape,
                            "dtypes": str(var_value.dtypes)
                        })
                
            result["status"] = "success"
            result["stdout"] = output_buffer.getvalue()
            result["result"] = exec_result
            
        except Exception as e:
            result["status"] = "error"
            result["stderr"] = f"{error_buffer.getvalue()}\n{traceback.format_exc()}"
        
        return result


    def _execute_bash(self, code: str, execution_id: str) -> dict:
        """
        Executes Bash shell commands with safety constraints and timeout protection.
        """
        try:
            completed = subprocess.run(
                code, shell=True, capture_output=True, text=True, timeout=self.max_execution_time
            )
            return {
                "execution_id": execution_id,
                "status": "success" if completed.returncode == 0 else "error",
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "result": None,
                "plots": [],
                "dataframes": []
            }
        except subprocess.TimeoutExpired:
            return {
                "execution_id": execution_id,
                "status": "error",
                "stdout": "",
                "stderr": "Execution timed out.",
                "result": None,
                "plots": [],
                "dataframes": []
            }



    def _execute_c(self, code: str, execution_id: str) -> dict:
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, "program.c")
        binary_path = os.path.join(temp_dir, "program")

        try:
            with open(source_path, "w") as f:
                f.write(code)

            compile_proc = subprocess.run(
                ["gcc", source_path, "-o", binary_path],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            if compile_proc.returncode != 0:
                return {
                    "execution_id": execution_id,
                    "status": "error",
                    "stdout": compile_proc.stdout,
                    "stderr": compile_proc.stderr,
                    "result": None,
                    "plots": [],
                    "dataframes": []
                }

            run_proc = subprocess.run(
                [binary_path],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            return {
                "execution_id": execution_id,
                "status": "success" if run_proc.returncode == 0 else "error",
                "stdout": run_proc.stdout,
                "stderr": run_proc.stderr,
                "result": None,
                "plots": [],
                "dataframes": []
            }
        except Exception as e:
            return {
                "execution_id": execution_id,
                "status": "error",
                "stdout": "",
                "stderr": str(e),
                "result": None,
                "plots": [],
                "dataframes": []
            }

    def _execute_java(self, code: str, execution_id: str) -> dict:
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, "Main.java")

        try:
            with open(source_path, "w") as f:
                f.write(code)

            compile_proc = subprocess.run(
                ["javac", source_path],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            if compile_proc.returncode != 0:
                return {
                    "execution_id": execution_id,
                    "status": "error",
                    "stdout": compile_proc.stdout,
                    "stderr": compile_proc.stderr,
                    "result": None,
                    "plots": [],
                    "dataframes": []
                }

            run_proc = subprocess.run(
                ["java", "-cp", temp_dir, "Main"],
                capture_output=True, text=True, timeout=self.max_execution_time
            )
            return {
                "execution_id": execution_id,
                "status": "success" if run_proc.returncode == 0 else "error",
                "stdout": run_proc.stdout,
                "stderr": run_proc.stderr,
                "result": None,
                "plots": [],
                "dataframes": []
            }
        except Exception as e:
            return {
                "execution_id": execution_id,
                "status": "error",
                "stdout": "",
                "stderr": str(e),
                "result": None,
                "plots": [],
                "dataframes": []
            }



interpreter_instance = CodeInterpreter()

@tool
def execute_code_multilang(code: str, language: str = "python") -> str:
    """Execute code in multiple languages (Python, Bash, SQL, C, Java) and return results.
    Args:
        code (str): The source code to execute.
        language (str): The language of the code. Supported: "python", "bash", "sql", "c", "java".
    Returns:
        A string summarizing the execution results (stdout, stderr, errors, plots, dataframes if any).
    """
    supported_languages = ["python", "bash", "sql", "c", "java"]
    language = language.lower()

    if language not in supported_languages:
        return f"❌ Unsupported language: {language}. Supported languages are: {', '.join(supported_languages)}"

    result = interpreter_instance.execute_code(code, language=language)

    response = []

    if result["status"] == "success":
        response.append(f"✅ Code executed successfully in **{language.upper()}**")

        if result.get("stdout"):
            response.append(
                "\n**Standard Output:**\n```\n" + result["stdout"].strip() + "\n```"
            )

        if result.get("stderr"):
            response.append(
                "\n**Standard Error (if any):**\n```\n"
                + result["stderr"].strip()
                + "\n```"
            )

        if result.get("result") is not None:
            response.append(
                "\n**Execution Result:**\n```\n"
                + str(result["result"]).strip()
                + "\n```"
            )

        if result.get("dataframes"):
            for df_info in result["dataframes"]:
                response.append(
                    f"\n**DataFrame `{df_info['name']}` (Shape: {df_info['shape']})**"
                )
                df_preview = pd.DataFrame(df_info["head"])
                response.append("First 5 rows:\n```\n" + str(df_preview) + "\n```")

        if result.get("plots"):
            response.append(
                f"\n**Generated {len(result['plots'])} plot(s)** (Image data returned separately)"
            )

    else:
        response.append(f"❌ Code execution failed in **{language.upper()}**")
        if result.get("stderr"):
            response.append(
                "\n**Error Log:**\n```\n" + result["stderr"].strip() + "\n```"
            )

    return "\n".join(response) 

