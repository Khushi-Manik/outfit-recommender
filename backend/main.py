if __package__:
    from .body_type_model import create_fastapi_app
else:
    from body_type_model import create_fastapi_app


app = create_fastapi_app()
