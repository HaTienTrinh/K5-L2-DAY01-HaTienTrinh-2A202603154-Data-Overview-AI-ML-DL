# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 6/11/2026 **

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ            | Nghĩa ngắn                                                    |
| ------------- | ------------------------------------------------------------- |
| Prediction    | Kết quả model vừa đoán ra; chưa phải nhãn đúng                |
| Ground truth  | Nhãn do người gán theo quy tắc                                |
| Taxonomy      | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint    | File model `.pt` đã huấn luyện sẵn                            |
| Guideline     | Quy tắc gán nhãn do người viết                                |
| Annotator     | Người gán nhãn                                                |
| Reviewer / QC | Người kiểm tra nhãn                                           |
| Escalation    | Ca khó → hỏi Lab Coach, không tự bịa                          |
| Model score   | Số “tin” của model; không phải điểm chất lượng nhãn người     |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   - Record này nói về cả bức ảnh vì: Record không có tọa độ box nào chỉ vào 1 vật thể
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp đi kèm taxonomy/checkpoint, model không tự nghĩ ra được
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - Cần ghi cả 3 vì mỗi cái trả lời 1 thông tin riêng
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Nếu ảnh có nhiều vật / nhiều chủ đề thì guideline phải nói rõ quy tắc chọn một lớp khi ảnh có nhiều chủ đề: chọn vật lớn nhất / chọn vật nhiều nhất / chọn vật ở giữa ảnh,...
6. Vì sao không được lấy `score` của model làm ground truth? - Score chỉ là mức tin của model vào dự đoán của chính nó, model vẫn có thể sai.
   vì score này khi thay đổi sẽ ảnh hưởng đến chất lượng của kết quả

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Góc trên bên trái: 385.33, 69.24
   - Góc dưới bên phải: 498.92, 348.92
   - Rộng: 113.58
   - Cao: 279.68
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - 0.20 có 17 prediction
   - 0.60 có 6 prediction
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Khi hạ ngưỡng giữ nhiều hộp hơn khiến người reviewer phải kiểm tra, rà soát lại do có thêm nhiều box hơn
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - Vẽ một box chữ nhật sát khít quanh phần nhìn thấy của vật, gồm toàn bộ vật thể và không chừa khoảng trống thừa; mỗi vật thật chỉ có đúng một box; không vẽ box cho vật không thuộc taxonomy của dự án.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - HÌnh bàn tay người chỉ hiện 1 nửa
   - Nếu trường hợp sảy ra thường xuyên thì sẽ viết thành quy tắc trong guideline, nếu guideline vẫn không nhận diện được thì sẽ đưa lên hỏi Lab Coach

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-008 ; `class_name`=spoon ; `score`=0.487579 ; số điểm polygon=`polygon_point_count`=46 ; 3–5 điểm đầu của `polygon_xy`=[
   544.0,
   67.0
   ],
   [
   544.0,
   80.0
   ],
   [
   545.0,
   81.0
   ],
   [
   545.0,
   91.0
   ]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - Box chỉ khoanh vật thể, còn polygon bám theo đường viền thật của vật thể cho biết hình dạng chính xác của vật thể
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - instance_id dùng để phân biệt từng cá thể trong cùng 1 ảnh.
   - Không phải class ID nghĩa là: nó không cho biết vật thuộc loại gì.
   - Không phải tracking ID nghĩa là: nó không cùng theo dõi 1 vật qua nhiều ảnh hoặc nhiều khung hình.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - Quy tắc vẽ biên mask: Chấm các điểm polygon dọc đúng theo mép ngoài của phần nhìn thấy của vật, đủ dày để bám sát các góc và đường cong (không cắt tròn thành góc vuông), không lấn sang nền hay vật lân cận, và mỗi vật có mask riêng, không gộp nhiều vật thành một.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - HÌnh bàn tay người chỉ hiện 1 nửa
   - Nếu là kiểu tình huống chung (hai vật chạm nhau thì mask chia ở đâu, vật bị che thì có vẽ phần khuất không) thì dùng guideline
   - Nếu là ca riêng không rõ (biên mờ tới mức không thể quyết định, vật khó nhận dạng) thì escalation lên Lab Coach, không tự bịa biên.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ                | Ground truth trông như thế nào? (đơn vị nhãn)                                             | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON                                                                                                                                  | Annotator làm gì?                                                                     | Reviewer xem gì?                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh         | Một (hoặc vài) class_name / class_id cho cả ảnh; đơn vị nhãn là ảnh                       | Ảnh traffic có nhiều chủ đề nhưng chỉ chọn một lớp; score 0.510915 của hạng 1 chưa chắc đúng; top-5 0.048194: nêu lớp hạng 2–3 nếu gần nghĩa hạng 1>> cho thấy lớp gần nhau dễ nhầm | Đọc guideline, chọn một lớp theo quy tắc chọn chủ đề chính; gắn cờ ảnh mơ hồ          | Lớp có đúng taxonomy không, có theo quy tắc chủ đề chính không, các ảnh tương tự có nhãn nhất quán không |
| Phát hiện vật thể     | Một box [x1, y1, x2, y2] + lớp cho mỗi vật; đơn vị nhãn là box                            | Vật bị che/cắt mép (tay); số box thay đổi theo ngưỡng <<0.60>>; box lệch hoặc trùng                                                                                                 | Vẽ box sát khít quanh phần nhìn thấy, chọn đúng lớp, không bỏ sót vật, không vẽ trùng | Box có sát không, đúng lớp không, có bỏ sót hoặc thừa box không, có tuân quy tắc che khuất không         |
| Instance segmentation | Một polygon (polygon_xy) + lớp + instance_id cho mỗi cá thể; đơn vị nhãn là mask từng vật | Biên mờ / hai vật chạm nhau; polygon quá thô hoặc lấn sang nền                                                                                                                      | Vẽ mask theo biên, tách từng cá thể riêng, theo quy tắc biên mask                     | Mask có khớp biên không, có gộp hoặc tách sai cá thể không, có phủ nền hoặc vật khác không               |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Chỉ dùng ảnh mẫu do giáo viên cung cấp cho bài lab, không tải thêm ảnh có người nhận diện được, không ghi họ tên, MSSV, email, số điện thoại hay dữ liệu cá nhân vào báo cáo/output (họ tên/MSSV chỉ nằm trong tên repository).
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi (có dữ liệu cá nhân, nội dung nhạy cảm, không thuộc bộ ảnh của bài lab), mình sẽ dừng ngay, không tiếp tục xử lý hay đăng lên GitHub, và báo cho Lab Coach / giáo viên của lớp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
