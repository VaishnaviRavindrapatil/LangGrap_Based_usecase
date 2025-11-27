import os
import re
import subprocess
from typing import TypedDict
from langgraph.graph import StateGraph
from openai import AzureOpenAI

# ----------------------------
# Azure OpenAI Configuration
# ----------------------------

client = AzureOpenAI(
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_key="",
    api_version="2024-02-01"
)

def get_gpt_response(prompt: str, model: str = "gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful software engineer assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# ----------------------------
# Language mappings
# ----------------------------
EXT_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby"
}
TEST_FRAMEWORK = {
    "python": "unittest",
    "javascript": "jest",
    "typescript": "jest",
    "java": "junit",
    "go": "go_test",
    "ruby": "rspec"
}

# ----------------------------
# Git Helpers (unchanged except env-based token optional)
# ----------------------------
def clone_github_repo(github_url: str, local_dir: str) -> str:
    repo_name = github_url.rstrip('/').split('/')[-1].replace('.git', '')
    repo_path = os.path.join(local_dir, repo_name)
    if not os.path.exists(repo_path):
        print(f"📥 Cloning {github_url} into {repo_path} ...")
        subprocess.run(["git", "clone", github_url, repo_path], check=True)
    else:
        print(f"✅ Repository already exists at {repo_path}")
    return repo_path

def git_commit_and_push(repo_path: str, branch_name: str, commit_message: str, github_token: str):
    os.chdir(repo_path)
    branches = subprocess.run(["git", "branch", "--list", branch_name], capture_output=True, text=True).stdout
    if branch_name not in branches:
        subprocess.run(["git", "checkout", "-b", branch_name], check=True)
    else:
        subprocess.run(["git", "checkout", branch_name], check=True)

    subprocess.run(["git", "add", "."], check=True)
    try:
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
    except subprocess.CalledProcessError:
        print("⚠️ Nothing new to commit.")
        return
    remote_url = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True).stdout.strip()
    if github_token and remote_url.startswith("https://"):
        token_url = remote_url.replace("https://", f"https://{github_token}@")
        subprocess.run(["git", "push", "-u", token_url, branch_name], check=True)
    else:
        subprocess.run(["git", "push", "-u", "origin", branch_name], check=True)
    print(f"✅ All changes pushed to '{branch_name}'")

# ----------------------------
# State Definition
# ----------------------------
class ProjectState(TypedDict):
    folder_path: str
    requirement: str
    recommendation: str
    generated_code: str
    test_file: str
    generated_tests: str
    language: str

# ----------------------------
# Helper: detect language by extension
# ----------------------------
def detect_language_for_file(fname: str) -> str:
    _, ext = os.path.splitext(fname)
    return EXT_LANG_MAP.get(ext.lower(), "unknown")

# ----------------------------
# Step 1: Analyze project (multi-language aware)
# ----------------------------
def analyze_project(inputs: ProjectState) -> ProjectState:
    folder_path = inputs["folder_path"]
    requirement = inputs["requirement"]

    top_py = any(f.endswith(tuple(EXT_LANG_MAP.keys())) for f in os.listdir(folder_path))
    if not top_py:
        subfolders = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        if len(subfolders) == 1:
            folder_path = os.path.join(folder_path, subfolders[0])
            print(f"📁 Auto-entered inner folder: {folder_path}")

    file_data = []
    file_candidates = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() not in EXT_LANG_MAP:
                continue
            if file.startswith("test_") or file.startswith("__") or file.endswith(".min.js"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    preview = f.read()[:1200]
            except:
                preview = ""
            lang = detect_language_for_file(file)
            file_data.append(f"File: {file} (lang: {lang})\nPreview:\n{preview}\n---")
            file_candidates.append({"name": file, "path": file_path, "lang": lang})

    prompt = f"""
Here are project files and their previews (various languages):

{chr(10).join(file_data)}

Requirement: {requirement}

Task:
- Choose exactly ONE file name (not path) where the implementation should be added or updated.
- Also output the language you expect to add code in.
Format of the single-line response (ONLY this):
FILENAME|LANGUAGE
Example: utils.js|javascript
"""
    resp = get_gpt_response(prompt).strip().strip("'\"")
    if "|" in resp:
        recommended_file, lang = [p.strip() for p in resp.split("|", 1)]
    else:
        recommended_file = resp
        # fallback: infer language by extension
        lang = detect_language_for_file(recommended_file)

    print(f"🧠 GPT recommends: {recommended_file} (language: {lang})")
    return {
        "folder_path": folder_path,
        "requirement": requirement,
        "recommendation": recommended_file,
        "generated_code": "",
        "test_file": "",
        "generated_tests": "",
        "language": lang
    }

# ----------------------------
# Step 2: Check if requirement already satisfied (per-language)
# ----------------------------
def check_existing_code(inputs: ProjectState) -> ProjectState:
    folder = inputs["folder_path"]
    file_name = inputs["recommendation"]
    requirement = inputs["requirement"]
    lang = inputs.get("language", "unknown")

    file_path = None
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower() == file_name.lower():
                file_path = os.path.join(root, f)
                break
        if file_path:
            break

    if not file_path:
        print(f"❌ File {file_name} not found. Will proceed to add code.")
        inputs["exists_ok"] = False
        return inputs

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except:
        content = ""

    prompt = f"""
You are a code analyst for {lang}.
File content preview:
{content[:3000]}

Requirement:
{requirement}

Question:
Does this file already contain code that fully satisfies the requirement?
Answer ONLY with one word: YES or NO.
"""
    answer = get_gpt_response(prompt).strip().upper()
    inputs["exists_ok"] = "YES" in answer
    print("✅ Requirement satisfied in file." if inputs["exists_ok"] else "🛠 Requirement not satisfied — will generate code.")
    return inputs

# ----------------------------
# Step 3: Add function (language-specific instructions)
# ----------------------------
def add_function(inputs: ProjectState) -> ProjectState:
    if inputs.get("exists_ok"):
        print("✅ Skipping addition (already satisfied).")
        return inputs

    folder_path = inputs["folder_path"]
    requirement = inputs["requirement"]
    file_name = inputs["recommendation"]
    lang = inputs.get("language", "python")

    target_path = None
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == file_name.lower():
                target_path = os.path.join(root, f)
                file_name = f
                break
        if target_path:
            break

    if not target_path:
        # create file in root with extension based on language
        ext = next((e for e, l in EXT_LANG_MAP.items() if l == lang), ".txt")
        target_path = os.path.join(folder_path, file_name if file_name.endswith(ext) else file_name + ext)
        open(target_path, "a").close()

    prompt = f"""
Requirement: {requirement}
Target language: {lang}
Target file: {file_name}

Instruction:
- Output ONLY the code to add (no explanations).
- Provide code appropriate for the file and language.
- If language is python, output a single def/function.
- If language is javascript/typescript, output an exported function (module.exports or export).
- Keep the snippet minimal and robust.
"""
    response = get_gpt_response(prompt)
    match = re.search(r"```(?:\w+)?\s*(.*?)```", response, re.DOTALL)
    code_snippet = (match.group(1).strip() if match else response.strip())

    with open(target_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + code_snippet + "\n")

    inputs["generated_code"] = code_snippet
    inputs["recommendation"] = file_name
    print(f"🧩 Added code to {file_name} (lang: {lang})")
    return inputs

# ----------------------------
# Step 4: Add tests (multi-language)
# ----------------------------

def add_tests(inputs: ProjectState) -> ProjectState:
    folder_path = inputs["folder_path"]
    file_name = inputs["recommendation"]
    lang = inputs.get("language", "python")
    generated_code = inputs.get("generated_code", "")

    # find the actual target file path (if present) so tests are created alongside it
    target_path = None
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() == file_name.lower():
                target_path = os.path.join(root, f)
                file_name = f
                break
        if target_path:
            break

    # use the directory of the target file; fall back to repo root
    target_dir = os.path.dirname(target_path) if target_path else folder_path

    # determine base module/name
    base, _ext = os.path.splitext(file_name)
    test_framework = TEST_FRAMEWORK.get(lang, "generic")

    # decide test file path per language, placed in target_dir
    if lang in ("javascript", "typescript"):
        ext_for_test = _ext or ".js"
        test_file = os.path.join(target_dir, f"{base}.test{ext_for_test}")
    elif lang == "java":
        test_file = os.path.join(target_dir, f"{base}Test.java")
    elif lang == "go":
        test_file = os.path.join(target_dir, f"{base}_test.go")
    else:
        test_file = os.path.join(target_dir, f"test_{base}.py")

    # determine function/class name heuristically
    m = re.search(r"(def|function|func|static\s+|public\s+).*?([A-Za-z_][A-Za-z0-9_]*)\s*\(", generated_code)
    func_name = m.group(2) if m else (base + "_func")

    prompt = f"""
Create tests for the {lang} function/class named '{func_name}' in module '{base}'.
Requirement: {inputs['requirement']}
Target test framework: {test_framework}

Instructions:
- Output ONLY the test file content, no explanation.
- Use idiomatic test style for the language/framework.
- Include valid and invalid cases.
"""
    test_content = get_gpt_response(prompt)
    match = re.search(r"```(?:\w+)?\s*(.*?)```", test_content, re.DOTALL)
    test_code = (match.group(1).strip() if match else test_content.strip())

    # ensure target directory exists (should, but be safe)
    os.makedirs(target_dir, exist_ok=True)

    #with open(test_file, "w", encoding="utf-8") as f:
    #    f.write(test_code + "\n")

    
    with open(test_file, "a", encoding="utf-8") as f:
        f.write("\n\n# ---- Auto-generated tests ----\n")
        f.write(test_code + "\n")

    inputs["test_file"] = test_file
    inputs["generated_tests"] = test_code
    print(f"✅ Test file created/updated: {test_file} (framework: {test_framework})")
    return inputs


# Step 5: Review and PR (unchanged)
# ----------------------------
def review_and_pr(inputs: ProjectState) -> ProjectState:
    if inputs.get("exists_ok"):
        print("✅ Requirement already satisfied. No further action needed.")
        return inputs

    folder_path = inputs["folder_path"]
    branch_name = TARGET_BRANCH
    repo_path = folder_path

    subprocess.run(["git", "checkout", branch_name], cwd=repo_path, check=False)
    subprocess.run(["git", "checkout", "-b", branch_name], cwd=repo_path, check=False)
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
    subprocess.run(["git", "commit", "-m", f"Auto-generated update for {inputs['recommendation']}"], cwd=repo_path, check=False)

    diff_result = subprocess.run(["git", "diff", "--stat", "HEAD~1", "--", "*"], capture_output=True, text=True, cwd=repo_path)
    diff_summary = diff_result.stdout.strip() or "No diff found."
    print("📄 Git diff summary:\n", diff_summary)

    review_prompt = f"""
You are a senior reviewer. Rate the changes 0-100 based on clarity, correctness, and tests.
Diff summary:
{diff_summary}
Output only a number.
"""
    score_response = get_gpt_response(review_prompt)
    try:
        score = int(re.search(r"\d+", score_response).group())
    except Exception:
        score = 0
    print(f"🧠 Review score: {score}%")

    if score >= 80:
        print("✅ Creating PR and pushing changes.")
        subprocess.run(["git", "push", "-u", "origin", branch_name], check=True, cwd=repo_path)
        pr_title = f"Auto-generated update: {inputs['recommendation']}"
        pr_body = f"Auto-generated code and tests for {inputs['recommendation']}"
        subprocess.run([r"C:\Program Files\GitHub CLI\gh.exe", "pr", "create", "--title", pr_title, "--body", pr_body, "--base", "main"], check=True, cwd=repo_path)
        merge_input = input("Merge PR into main? (yes/no): ").strip().lower()
        if merge_input in ("yes", "y"):
            subprocess.run([r"C:\Program Files\GitHub CLI\gh.exe", "pr", "merge", branch_name, "--merge"], check=True, cwd=repo_path)
            print("✅ PR merged.")
    else:
        print("⚠️ Score below threshold; PR not created.")

    return inputs

# ----------------------------
# Build LangGraph with multi-language flow
# ----------------------------
graph = StateGraph(ProjectState)
graph.add_node("analyze_project", analyze_project)
graph.add_node("check_existing_code", check_existing_code)
graph.add_node("add_function", add_function)
graph.add_node("add_tests", add_tests)
graph.add_node("review_and_pr", review_and_pr)

graph.set_entry_point("analyze_project")
graph.add_edge("analyze_project", "check_existing_code")
graph.add_conditional_edges(
    "check_existing_code",
    lambda s: "skip" if s.get("exists_ok") else "continue",
    {"skip": "__end__", "continue": "add_function"}
)
graph.add_edge("add_function", "add_tests")
graph.add_edge("add_tests", "review_and_pr")
graph.set_finish_point("review_and_pr")

# ----------------------------
# Run Flow (example)
# ----------------------------
GITHUB_URL = "https://github.com/VaishnaviRavindrapatil/Test_Java_Files.git"
#GITHUB_URL = "https://github.com/VaishnaviRavindrapatil/Test_Python_File.git"
LOCAL_BASE_DIR = r"C:\Users\vaishnaviravindra_pa\Documents"
GITHUB_TOKEN = "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"

TARGET_BRANCH = input("Enter the branch name to push changes: ")

repo_path = clone_github_repo(GITHUB_URL, LOCAL_BASE_DIR)
inputs: ProjectState = {
    "folder_path": repo_path,
    "requirement": "add function who will remove capital characters from string",
    "recommendation": "",
    "generated_code": "",
    "test_file": "",
    "generated_tests": "",
    "language": "python"
}

app = graph.compile()
result = app.invoke(inputs)
print("✅ Flow completed. Recommendation:", result.get("recommendation"))