PYTHONPATH=src
CONFIG=configs/default.yaml

.PHONY: audit preprocess preprocess-blue-green preprocess-rod preprocess-rod-blue prepare-rod prepare-rod-blue download-cat train train-conservative train-third train-fourth train-blue-green-second train-rod train-rod-blue eval infer export review alice-preview alice-fourth-preview eval-third review-third wait-third eval-fourth review-fourth run-fourth run-blue-green-second run-rod run-rod-blue eval-rod eval-rod-blue review-rod review-rod-blue download-rod-weights top-success-fourth eval-blue-green-second review-blue-green-second supervised-ablation-report threshold-sweep generate-controlled-ablations run-controlled-ablations gpu-benchmark cpu-robot-benchmark

audit:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config $(CONFIG) audit

preprocess:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config $(CONFIG) preprocess

preprocess-blue-green:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/blue_green_second_run.yaml preprocess

download-cat:
	bash scripts/download_cat.sh

preprocess-rod:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat.yaml audit
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat.yaml preprocess

prepare-rod: download-cat download-rod-weights preprocess-rod

preprocess-rod-blue:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat_blue.yaml audit
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat_blue.yaml preprocess

prepare-rod-blue: download-cat download-rod-weights preprocess-rod-blue

train:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config $(CONFIG) train

train-conservative:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/conservative.yaml train

train-third:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/third_run.yaml train

train-fourth:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/fourth_run.yaml train

train-blue-green-second:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/blue_green_second_run.yaml train

download-rod-weights:
	bash scripts/download_efficient_sam_vits.sh

train-rod:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat.yaml train

eval-rod:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat.yaml eval

review-rod:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat.yaml review --split test --top-k 10

run-rod:
	bash scripts/run_train_eval_review.sh configs/rod_vits_cat.yaml

train-rod-blue:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat_blue.yaml train

eval-rod-blue:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat_blue.yaml eval

review-rod-blue:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/rod_vits_cat_blue.yaml review --split test --top-k 10

run-rod-blue:
	bash scripts/run_train_eval_review.sh configs/rod_vits_cat_blue.yaml

eval:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config $(CONFIG) eval

infer:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config $(CONFIG) infer --input-dir CAT/mixed/Test/imgs --output-dir outputs/infer

export:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config $(CONFIG) export

review:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config $(CONFIG) review --split test --top-k 10

eval-third:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/third_run.yaml eval

eval-fourth:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/fourth_run.yaml eval

eval-blue-green-second:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/blue_green_second_run.yaml eval

review-third:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/third_run.yaml review --split test --top-k 10

review-fourth:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/fourth_run.yaml review --split test --top-k 10

review-blue-green-second:
	PYTHONPATH=$(PYTHONPATH) python -m ttfm.cli --config configs/blue_green_second_run.yaml review --split test --top-k 10

run-blue-green-second:
	bash scripts/run_train_eval_review.sh configs/blue_green_second_run.yaml

wait-third:
	bash scripts/wait_then_postprocess.sh configs/third_run.yaml

run-fourth:
	bash scripts/run_train_eval_review.sh configs/fourth_run.yaml

top-success-fourth:
	PYTHONPATH=$(PYTHONPATH) python scripts/top_success_examples.py --config configs/fourth_run.yaml --split test --top-k 20

supervised-ablation-report:
	PYTHONPATH=$(PYTHONPATH) python scripts/supervised_ablation_report.py

threshold-sweep:
	PYTHONPATH=$(PYTHONPATH) python scripts/threshold_sweep.py --config configs/blue_green_second_run.yaml --split test

generate-controlled-ablations:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_controlled_ablations.py --generate-only

run-controlled-ablations:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_controlled_ablations.py

gpu-benchmark:
	PYTHONPATH=$(PYTHONPATH) python scripts/gpu_inference_benchmark.py --config configs/blue_green_second_run.yaml --batch-size 8 --limit 128 --repeats 20

cpu-robot-benchmark:
	PYTHONPATH=$(PYTHONPATH) python scripts/gpu_inference_benchmark.py --config configs/blue_green_second_run.yaml --device cpu --batch-size 1 --limit 128 --repeats 10 --output reports/cpu_robot_inference_benchmark.json

alice-preview:
	PYTHONPATH=$(PYTHONPATH) python scripts/alice_preview.py

alice-fourth-preview:
	PYTHONPATH=$(PYTHONPATH) python scripts/alice_fourth_preview.py
