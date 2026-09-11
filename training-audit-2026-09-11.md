# Đánh giá huấn luyện — 11/09/2026

**Cập nhật sau khi kiểm tra lại `edge_asr/`:** dữ liệu mới nhất vẫn là run `qi1866j3`, bắt đầu 10/09/2026, kết thúc ở 19.896 update; chưa có log chạy notebook đã sửa. Người dùng xác nhận môi trường sử dụng trước mắt là **GPU Kaggle**. Vì vậy, phương án ưu tiên là tiếp tục kiểm chứng ConMamba trên Kaggle; chưa cần đổi kiến trúc vì ONNX hoặc thiết bị ARM.

**Phương án cho phiên Kaggle tiếp theo:**

| Bước | Cách thực hiện | Điều kiện đi tiếp |
|---|---|---|
| 1. Chốt baseline | Load `retokenized_continue_best.pt`, kiểm tra tokenizer/split, chạy full validation 2.000 mẫu với greedy; lưu dự đoán, lỗi xóa/thay/chèn theo ngôn ngữ, nguồn và độ dài | Macro-CER xấp xỉ 46,8486% trong cùng điều kiện; nếu lệch đáng kể, giải thích sai khác trước khi train |
| 2. Sửa recipe optimizer | Chia nhóm theo cả LR và weight decay; tôn trọng `_no_weight_decay` của Mamba. Dùng stage/tag mới và optimizer mới từ trọng số best | Đã triển khai; smoke phải pass trước khi chạy Kaggle và không nạp optimizer state cũ vào cấu trúc nhóm mới |
| 3. So sánh hai nhánh ngắn | Mỗi nhánh 1 epoch, cùng checkpoint nguồn, seed, sampler, batch 32, LR backbone `1e-5`, head `1e-4`, warmup 100 update; A: CTC + KD `0.05`; B: chỉ CTC | Chọn bằng full validation macro-CER và CER từng ngôn ngữ, không chọn theo train loss |
| 4. Quyết định ngân sách | Chỉ kéo dài nhánh có cải thiện rõ so với baseline; theo dõi thêm tối đa 1–2 epoch với lịch LR được cấu hình từ trước | Không tự đặt lại cosine mỗi phiên hoặc tăng số epoch trong im lặng; lưu baseline nếu mọi nhánh kém hơn |
| 5. Decoder và test | Tune greedy/prefix beam trên validation, tính cả thời gian decoder; khóa checkpoint + decoder rồi mới chạy test | Có CER/WER test và RTF/latency trên một GPU Kaggle, phân biệt forward-only với end-to-end |

Các LR và warmup trong bảng là **cấu hình thử nghiệm đề xuất**, chưa được kiểm chứng. Notebook hiện có hai stage enabled `retokenized_continue_kd_probe` và `retokenized_continue_ctc_probe`, cùng nạp `retokenized_continue_best.pt`, mỗi stage 1 epoch, với optimizer mới và tag riêng. Hai nhánh cần dùng cùng recipe đã sửa để chỉ khác trọng số KD; nếu tốt lên so với baseline vẫn chưa thể quy toàn bộ mức tăng cho một thay đổi riêng lẻ. Ngân sách ban đầu đề xuất khoảng 2–3 giờ GPU cho hai nhánh dựa trên lần chạy cũ, cần điều chỉnh bằng thời gian thực đo; không mặc định dùng hết quota tuần. Nếu chênh lệch macro-CER nhỏ, nên xác nhận bằng seed khác hoặc bootstrap theo source/speaker trước khi kết luận.

**Lý do điều chỉnh phương án:**

- Không có dấu hiệu phải reset CTC head hoặc train từ đầu: mô hình không collapse và checkpoint continuation cải thiện so với joint best. Tokenizer audit ghi 0 unknown và 0 byte-fallback pieces trên tập train đã audit của cả tám ngôn ngữ; chưa có bằng chứng cần thay tokenizer lần nữa. Kết quả này không chứng minh coverage của validation/test hay tiếng nói thực tế.
- KD validation gần như đứng ở 0,373–0,377, trong khi CTC loss giảm. Điều đó đủ để đặt câu hỏi KD còn giúp continuation hay không, chưa đủ để khẳng định KD gây hại. Hai nhánh A/B trả lời trực tiếp câu hỏi này.
- Optimizer cũ dùng `AdamW(..., weight_decay=0.01)` cho mọi tham số trong từng nhóm LR, bỏ qua cờ mà Mamba gắn cho `A_log` và `D`. Recipe probe mới đã tạo nhóm `weight_decay=0` cho đúng các tham số có `_no_weight_decay`; chưa có bằng chứng đây là nguyên nhân chính của CER cao. Đối chiếu [mã nguồn Mamba](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py).
- Không tăng batch/worker trước: log cũ có batch 32, smoke 0,58 giây/step và chờ data gần 0 ở độ chính xác hiển thị. Sửa đọc cache, kiểm tra CTC và validation ở lượt trước cần được đo lại trước khi thay điều kiện tối ưu hóa.
- Bộ trọng số có **33.372.481 tham số**, khoảng 133,5 MB nếu tính 4 byte/tham số, phù hợp hướng thử nghiệm GPU Kaggle. File latest khoảng 401 MB có cả trạng thái optimizer nên không phải kích thước mô hình inference. Có thể bỏ `kl_head` khi làm inference-only, nhưng chỉ tiết kiệm khoảng 329 nghìn tham số (~1,3 MB FP32), không giải quyết phần lớn footprint.

**Giới hạn triển khai còn tồn tại:** inference trên từng đoạn audio với GPU Kaggle là mục tiêu gần phù hợp. Streaming giữ state liên tục chưa được xây dựng: notebook gọi `self.mamba(x)` mà không truyền cache, và CNN/ConMamba convolution chưa có bộ đệm liên chunk. Mamba upstream có `step` và cache nhưng cần tích hợp xuyên suốt encoder; đường `step` hiện kiểm tra đầu vào một frame mỗi lần. Ngoài ra, mel transform trong cell tạo cache không đặt `center=False`; với `n_fft=400`, 16 kHz, STFT centered có nửa cửa sổ khoảng 12,5 ms cần được tính vào cách đệm/độ trễ khi chuyển sang streaming. Không đổi preprocessing của checkpoint hiện có mà không kiểm tra sự tương đương. Tham khảo [Mamba inference](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) và [MelSpectrogram](https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html).

Với mục tiêu hiện tại, **chưa nên chuyển sang backbone khác hoặc ưu tiên export ONNX**. Nếu sau này cần ONNX/CPU, cần kiểm chứng đường export và runtime trước khi đầu tư thêm training; custom operator có thể cần translation/decomposition riêng, không nên coi export là thao tác một lệnh. Tham khảo [hướng dẫn mở rộng ONNX exporter](https://docs.pytorch.org/tutorials/beginner/onnx/onnx_registry_tutorial.html). Lượt đánh giá lại này cập nhật phương án, không đổi notebook thêm và không khởi chạy job Kaggle.

Mô hình đang học được nhưng chất lượng nhận dạng còn thấp. Lần chạy ngày 10/09 hoàn tất bình thường, không có bằng chứng CTC collapse. Checkpoint nên dùng cho đánh giá tiếp theo là `edge_asr/checkpoints/retokenized_continue_best.pt`: epoch 3, update 14.922, macro-CER validation **46,85%**. Chưa có kết quả test hoặc benchmark streaming mới trong những log được kiểm tra.

## Bằng chứng từ lần chạy hiện tại

Nguồn chính: `edge_asr/training_log.jsonl`, `wandb/run-20260910_041341-qi1866j3/files/output.log`, `wandb-summary.json`, metadata trong các checkpoint `*_best.pt` và ba manifest split. File `knowledge-distillation-experiment.log` là log pseudo-label cũ; cuối file ghi 28.600 mẫu đã lưu, 0 lỗi ở mốc đó, cùng nhiều cảnh báo `max_new_tokens`/`max_length`. Nó không mô tả lần train tháng 9 và không đủ để kết luận job tạo nhãn đã hoàn thành.

Luồng thực tế hiện tại khác mô tả cũ trong AGENTS.md: cache chunk teacher/mel → khôi phục group split → tokenizer chuẩn hóa train-only 8.000 token (+ blank) → retokenize target → ConMamba → smoke và CTC preflight → `retokenized_continue` từ `retokenized_joint_best`. Front-end CNN trong notebook hiện đã dùng convolution causal. Cell audit checkpoint/test/diagnostic cuối notebook đang tắt.

| Epoch continuation | Train loss | Val loss | WER | CER | Macro-CER |
|---|---:|---:|---:|---:|---:|
| 1 | 2,8978 | 2,8983 | 75,32% | 46,40% | 49,94% |
| 2 | 2,6989 | 2,8093 | 74,39% | 45,13% | 48,53% |
| **3 — best** | **2,5228** | **2,7440** | **73,04%** | **43,97%** | **46,85%** |
| 4 | 2,4333 | 2,7380 | 73,16% | 43,97% | 46,93% |

- Checkpoint nguồn `retokenized_joint_best` có macro-CER 49,8292%; continuation best đạt 46,8486%, cải thiện **2,98 điểm phần trăm** trên cùng split. Loss thấp nhất không đồng nghĩa checkpoint tốt nhất theo macro-CER.
- Có 19.896 optimizer updates thành công, 8 AMP scaler skips trên 19.904 lần thử (~0,04%), không ghi nhận non-finite loss hoặc collapse. Scaler skips riêng lẻ chưa cho thấy mất ổn định kéo dài.
- Khoảng thời gian từ `session_start` tới `training_complete` là **4 giờ 18 phút**, gồm cả phần chuẩn bị trong log; không phải 8 giờ hết quota. LR cuối khoảng `1,26e-11`, phù hợp cuối cosine schedule.
- Smoke cũ: median **0,58 giây/step**, thời gian chờ data gần 0 ở độ chính xác được in; peak **5,74 GB** trên GPU đang được đo. Log cũ chưa ghi peak từng GPU, không nên dùng số memory sau backward ~1 GB để suy ra có thể tăng batch nhiều lần.
- Train/validation/test có **139.539 / 2.000 / 2.000** mẫu. Tập chỉ số trong `splits.json` và manifest retokenized trùng khớp hoàn toàn. Lỗi smoke tạo lại split chưa làm lệch lần chạy này, nhưng có thể gây lệch khi retokenization lọc bỏ thêm mẫu.

CER cuối epoch 4 theo ngôn ngữ: Đức 30,67%; Việt 32,67%; Tây Ban Nha 33,03%; Pháp 42,59%; Anh 54,26%; Hàn 55,51%; Trung 62,78%; Nhật 63,92%. Đây là số của **epoch cuối**, không phải per-language breakdown của epoch best. Blank argmax khoảng 85%, nhưng hypothesis rỗng chỉ 0,1%; không nên coi blank cao đơn lẻ là collapse. Số token dự đoán trung bình ~19,8 so với tham chiếu ~26,4 gợi ý cần đo lỗi xóa/thay/chèn; chưa đủ để khẳng định nguyên nhân là deletion.

## Những sửa đổi trong notebook

1. **Bảo toàn split và smoke gate.** Smoke dùng trực tiếp dataset/split đã audit, kiểm tra chỉ số theo manifest, không tạo `splits.json` khác. Reset trạng thái pass trước mọi kiểm tra có thể lỗi; yêu cầu đủ 50 batch; đo peak mọi GPU. Giải phóng tensor/graph diagnostic trước training.
2. **Resume learning rate.** Lưu dữ liệu cosine schedule trong state scheduler; resume bình thường giữ nguyên đường LR, không tự đặt sàn 1/3 và dựng lại cosine mỗi phiên. Chỉ cho phép sàn khi chủ động kéo dài một kế hoạch đã hoàn tất. Checkpoint cũ thiếu dữ liệu schedule được đối chiếu với LR thực; nếu không thể tái dựng lịch cũ, nối đuôi liên tục từ LR đã lưu. Điều này xử lý hạn chế của [LambdaLR: Python closure không được lưu tự động](https://docs.pytorch.org/docs/main/generated/torch.optim.lr_scheduler.LambdaLR.html).
3. **Resume data và accumulation.** Bỏ batch đã tiêu thụ ở sampler trước khi worker đọc/giải nén NPZ. Xử lý cửa sổ gradient accumulation cuối epoch khi số batch không chia hết GA; lưu batch size/GA thực của stage. Giữ cập nhật scaler ở ranh giới accumulation theo [hướng dẫn AMP của PyTorch](https://docs.pytorch.org/docs/main/notes/amp_examples.html#gradient-accumulation).
4. **Validation và checkpoint selection.** Full validation không còn bị cắt bởi batch cap. Nếu không đủ thời gian chạy validation cuối epoch, lưu cursor chờ validation và trả về paused; phiên sau đánh giá rồi mới tăng epoch. Không tính checkpoint collapse/non-finite là cải thiện. Khi bắt đầu continuation mới, đo và giữ baseline nguồn trên full validation để model không mặc nhiên thay bằng một epoch kém hơn.
5. **Metric và audit.** Phân biệt loss với macro-CER trong thông báo hoàn tất và summary; sửa dòng cũ in `macro_cer=2.743953`. Log validation thêm scope, sample count, collapse và per-language metrics. Resume chỉ để validation không còn ghi train loss giả bằng 0. Preflight cho phép resume từ continuation khi checkpoint nguồn upstream không còn gắn vào notebook.
6. **Tối ưu tính toán có bảo toàn kết quả.** Vector hóa số frame tối thiểu của CTC, bao gồm token lặp; lấy vector lengths về CPU một lần cho mỗi phía KD thay vì `.item()` từng mẫu. Greedy validation tính argmax trên GPU và chỉ chuyển ID về CPU: với vocab 8.001, lượng dữ liệu chuyển cho logits giảm khoảng 4.000 lần; đây là giảm số byte, **không phải cam kết tốc độ validation tăng 4.000 lần**. Tiny overfit dùng GradScaler như training.

Giữ nguyên fp32 island của ConMamba, cosine KD và nội suy theo từng utterance, cơ chế gradient checkpointing ở cấp class, kiến trúc, tokenizer, LR khởi điểm, số epoch và effective batch 32. Không chỉnh checkpoint hay log lịch sử. Không bật job GPU hoặc đánh giá test tự động.

## Bước tiếp theo trên Kaggle

1. Dùng notebook đã sửa, chạy lại các cell định nghĩa/configuration trước smoke; định nghĩa training mới là `scratch-quality-throughput-v8`. Gắn cache và checkpoint continuation hiện có. Recipe 4 epoch đã hoàn tất sẽ được nhận diện là complete; rerun không tự kéo dài training.
2. Audit checkpoint best, lấy per-language CER và breakdown xóa/thay/chèn trên validation, ưu tiên Nhật/Trung/Hàn và Anh. Soát chất lượng transcript, ranh giới chunk và mức độ thiếu token tương ứng. Chỉ điều chỉnh sampling/nhãn khi có bằng chứng; hiện French/Spanish/German được sampler lặp 2 lần, các ngôn ngữ còn lại 1 lần.
3. Thử/tune decoder trên validation, khóa cấu hình rồi đánh giá held-out test. Không chọn checkpoint hoặc decoder bằng test. Cần phân biệt CER và WER khi so sánh các hệ chữ không phân tách từ bằng khoảng trắng.
4. Đo lại throughput và peak từng GPU với bản sửa. Giữ batch 32 trước; log smoke cũ không cho thấy nghẽn DataLoader. Chỉ cân nhắc tăng batch hoặc đổi DataParallel sau phép đo, vì thay effective batch cũng thay điều kiện tối ưu hóa.
5. Nếu vẫn cần train thêm, tạo continuation có tag riêng từ **continuation best**, có baseline và ngân sách giới hạn. Việc epoch 4 train loss tiếp tục giảm trong khi macro-CER gần như đứng yên chưa đủ để kết luận overfit nặng, nhưng không ủng hộ nối thêm nhiều epoch với cùng recipe mà không phân tích lỗi.

## Kiểm chứng và giới hạn

Đã kiểm tra JSON và AST của toàn bộ 22 code cell (các lệnh IPython shell được loại khỏi bước AST), đối chiếu thay đổi với bản notebook đầu phiên và giữ nguyên các cell ngoài phạm vi sửa. Cả **8 kiểm thử CPU đã pass** với PyTorch 2.10.0+cpu: decoder giữ blank/repeat/padding semantics; số frame CTC tương đương bản cũ; giá trị và gradient KD trùng khớp bản cũ; lịch LR qua save/load; bỏ batch trước đọc dữ liệu; accumulation dư và metric summary; pause trước validation rồi resume không cập nhật lại trọng số; stage hoàn tất không train lại. Ba kiểm thử control flow dùng model/loss/validation giả lập tối giản, không thay thế kiểm thử ConMamba GPU. AST cũng được kiểm tra theo ngữ pháp Python 3.12 của Kaggle. CUDA selective-scan, DataParallel, peak T4, tốc độ và chất lượng sau sửa phải được xác nhận bằng smoke/evaluation trên Kaggle; chưa có kết quả GPU mới.
