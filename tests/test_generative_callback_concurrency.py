import ast
from pathlib import Path


MODULE = Path(__file__).parents[1] / "gas/evaluation/generative_challenge_manager.py"


def _manager_class():
    tree = ast.parse(MODULE.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerativeChallengeManager"
    )


def test_callback_stages_then_submits_background_processing():
    manager = _manager_class()
    callback = next(
        node
        for node in manager.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "generative_callback"
    )
    calls = [node for node in ast.walk(callback) if isinstance(node, ast.Call)]

    assert any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "run_in_executor"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Attribute)
        and call.args[1].attr == "_write_staged_callback"
        for call in calls
    )
    assert any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "submit"
        and call.args
        and isinstance(call.args[0], ast.Attribute)
        and call.args[0].attr == "_process_staged_callback"
        for call in calls
    )
    responses = [
        call for call in calls
        if isinstance(call.func, ast.Name) and call.func.id == "Response"
    ]
    assert any(
        any(
            keyword.arg == "status_code"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == 202
            for keyword in call.keywords
        )
        for call in responses
    )


def test_storage_pipeline_is_synchronous_for_bounded_executor():
    manager = _manager_class()
    storage = next(node for node in manager.body if node.name == "store_binary_content")
    assert isinstance(storage, ast.FunctionDef)
