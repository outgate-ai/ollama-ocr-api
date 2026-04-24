"""CLI entrypoint: ``ocr-api`` or ``python -m ocr_api``.

Configuration precedence: CLI flag > environment variable > built-in default.
See ENVIRONMENT.md for the full list of supported variables.
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .config import Config
from .server import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ocr-api",
        description="Ollama-compatible HTTP server for vision/OCR models.",
    )
    parser.add_argument("--host", default=None, help="Host to bind (env: OCR_API_HOST)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind (env: OCR_API_PORT)")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=None,
        help="Inference device (env: OCR_API_DEVICE)",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default=None,
        help="Model dtype (env: OCR_API_DTYPE)",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face repo id for the vision model (env: OCR_API_MODEL_ID)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Name advertised in /api/tags and required on /api/chat (env: OCR_API_MODEL_NAME)",
    )
    parser.add_argument(
        "--default-prompt",
        default=None,
        help="Prompt used when a request omits content (env: OCR_API_DEFAULT_PROMPT)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Default max generation tokens when options.num_predict is missing "
        "(env: OCR_API_MAX_NEW_TOKENS)",
    )
    parser.add_argument(
        "--max-image-bytes",
        type=int,
        default=None,
        help="Reject any single decoded image larger than this (env: OCR_API_MAX_IMAGE_BYTES)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level: debug|info|warning|error (env: OCR_API_LOG_LEVEL)",
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="If set, /api/* endpoints require Authorization: Bearer <token> or "
        "X-API-Key: <token> (env: OCR_API_AUTH_TOKEN). Unset means auth disabled.",
    )
    args = parser.parse_args(argv)

    cfg = Config.from_env().override(
        host=args.host,
        port=args.port,
        device=args.device,
        dtype=args.dtype,
        model_id=args.model_id,
        model_name=args.model_name,
        default_prompt=args.default_prompt,
        max_new_tokens_default=args.max_new_tokens,
        max_image_bytes=args.max_image_bytes,
        log_level=args.log_level,
        auth_token=args.auth_token,
    )

    logging.basicConfig(
        level=cfg.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_app(
        model_id=cfg.model_id,
        model_name=cfg.model_name,
        device=cfg.device,
        dtype=cfg.dtype,
        default_prompt=cfg.default_prompt,
        max_new_tokens_default=cfg.max_new_tokens_default,
        max_image_bytes=cfg.max_image_bytes,
        auth_token=cfg.auth_token,
    )
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level=cfg.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
