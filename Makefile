format:
	poetry run black nlinprog/
	poetry run isort nlinprog/
	
	poetry run black tests/
	poetry run isort tests/
	
lint: 
	poetry run ruff check nlinprog/. --fix
	poetry run ruff check tests/. --fix

test:
	poetry run pytest tests/.