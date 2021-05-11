
test:
	python -m unittest discover -s tests -p 'test_*.py'

test-short:
	SHORT=True python -m unittest discover -s tests -p 'test_*.py'

