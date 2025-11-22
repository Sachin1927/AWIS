api:
	@echo "Starting FastAPI backend..."
	cd src/api && python main.py

models:
	@echo "Training ML models..."
	python src/ml/train_attrition.py
	python src/ml/train_forecast.py
	python src/ml/train_mobility.py