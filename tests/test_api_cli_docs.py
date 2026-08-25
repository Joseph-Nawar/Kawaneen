from __future__ import annotations


def test_api_cli_defaults_and_openapi_contract() -> None:
    from kawaneen.api.app import create_app
    from kawaneen.cli import build_parser

    args = build_parser().parse_args(["api", "serve"])
    assert args.api_command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8000

    schema = create_app().openapi()
    required = {
        "/v1/search",
        "/v1/answer",
        "/v1/extract",
        "/v1/documents",
        "/v1/documents/{document_id}",
        "/v1/health",
        "/v1/models",
    }
    assert required <= set(schema["paths"])
    operation_ids = [
        operation["operationId"]
        for path in schema["paths"].values()
        for operation in path.values()
        if isinstance(operation, dict) and "operationId" in operation
    ]
    assert len(operation_ids) == len(set(operation_ids))
