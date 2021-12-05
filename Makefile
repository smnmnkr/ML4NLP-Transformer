module := transformer

#
# run demo:
install:
	@python3 -m pip install -r requirements.txt

run:
	@python3 -m ${module}
