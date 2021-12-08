module := transformer
config := config.json

#
# run demo:
install:
	@python3 -m pip install -r requirements.txt
	@python3 -m pip install -U spacy
	@python3 -m spacy download en_core_web_sm
	@python3 -m spacy download de_core_news_sm

run:
	@python3 -m ${module} -C ${config}
