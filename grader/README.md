# Task 1: chấm điểm rubric bài C++ bằng Qwen2.5-1.5B-Instruct (LoRA)

Pipeline tự động hoàn toàn, suy luận greedy (`do_sample=False`), seed cố định 42. Input gồm đề bài, cấu trúc điểm và chính sách của đề, `compile_log`, `test_report`, code. Không có trường `feedback` nào trong input.

## Cài đặt

    pip install -r requirements.txt

## Chạy

    # 1. Huấn luyện (--val_ratio 0.2 để giữ tập kiểm định; bản nộp cuối dùng 0)
    python train.py --train_root ../data/sample_dataset --output_dir ../runs/v1 --val_ratio 0.2
    # 2. Suy luận tập test, ra submission_task1.json
    python infer.py --test_root ../data/public_test_release --adapter_dir ../runs/v1 --output_dir ../runs/v1/preds
    # 3. Đánh giá trên mẫu giữ lại (QWK, MAE, exact-match từng chiều)
    python infer.py --test_root ../data/sample_dataset --adapter_dir ../runs/v1 --output_dir ../runs/v1/val_preds --with_labels
    python evaluate.py --data_root ../data/sample_dataset --predictions_dir ../runs/v1/val_preds --adapter_dir ../runs/v1

## Tái lập

`run_manifest.json` ghi model id, commit hash model, seed, siêu tham số, mẫu giữ lại, phiên bản thư viện. `infer.py` tải đúng revision đó.

## Chính sách chấm (policy.py)

Áp dụng tất định sau khi model sinh điểm, nên tổng luôn bằng tổng 6 chiều:
- Không biên dịch được (`compilable = 0`): mọi chiều về 0.
- Đề nhiều câu: câu tiên quyết (đọc từ `is_prerequisite_for` trong `exams.json`) chưa làm (thiếu hàm, thân rỗng, hoặc chỉ `return`) thì `logic`, `edge_case`, `complexity`, `code_quality` về 0.

Kiểm thử: `python -m pytest -q tests`
