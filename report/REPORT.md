# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** T4 GPU

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không đổi checkpoint, ngưỡng hay mã xử lý. Chỉ thêm một ô phụ để nén và tải
`day1_lab_outputs/` về máy, phục vụ việc đọc JSON khi viết báo cáo.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=`cab` ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=`ImageNet-1K`
2. Record này nói về **cả bức ảnh**. Record không có `bbox_xyxy` hay `polygon_xy`, chỉ có `image_width`/`image_height`
   của toàn ảnh và một danh sách xếp hạng (rank 1–5: cab, minibus, police_van, recreational_vehicle, streetcar).
   Model không chỉ ra chiếc taxi nằm ở đâu. Nó chỉ nói “ảnh này giống lớp `cab` nhất”.
3. Danh sách lớp **đi kèm taxonomy/checkpoint**: `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ có thể trả
   về một trong 1000 lớp đó. Model không tự nghĩ ra tên lớp mới. Ví dụ ảnh `kitchen` bị đoán là `gong` (0.42) vì
   ImageNet-1K không có lớp “kitchen”. Model buộc phải chọn lớp gần nhất trong danh sách.
4. Tên lớp có thể trùng hoặc khác nghĩa giữa các taxonomy. `class_id` 468 chỉ có nghĩa là `cab` **trong** ImageNet-1K.
   Trong COCO-80, id 5 là `bus`. Ghi cả ba trường giúp truy ngược đúng danh sách lớp, tránh nhầm khi gộp dữ liệu
   từ nhiều nguồn và cho phép máy đọc bằng id thay vì so sánh chuỗi tên.
5. Guideline phải nêu rõ **quy tắc chọn lớp chính**. Ví dụ: chọn chủ thể chiếm diện tích lớn nhất hoặc nằm ở trung tâm,
   hoặc theo thứ tự ưu tiên lớp đã định sẵn. Guideline cũng phải nói phải làm gì khi không có lớp nào phù hợp
   (gán `other` hay escalation). Ảnh `traffic` có cả taxi, xe buýt và xe tải. Nếu không có quy tắc, mỗi annotator sẽ chọn một kiểu.
6. `score` chỉ là mức tự tin của model, không phải sự thật. Model có thể sai mà vẫn tự tin. Ví dụ ảnh `kitchen` →
   `gong` 0.42 là sai rõ ràng. Hơn nữa score 0.51 cũng chỉ hơn một nửa. Ground truth phải do người gán theo guideline
   và được reviewer kiểm tra. Nếu lấy prediction làm nhãn, lỗi của model sẽ được đưa ngược lại vào dữ liệu huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `class_name`=`person` ; `score`=0.913 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Ảnh rộng 640 × cao 427 pixel, gốc tọa độ (0,0) ở góc trên-trái và trục y hướng xuống. Góc trên-trái của hộp ở
   (x≈385, y≈69), tức hơi lệch phải và gần mép trên, ngang đầu người đầu bếp. Góc dưới-phải ở (x≈499, y≈349),
   ngang bàn chân. Hộp rộng ≈114 px (= 498.92 − 385.33) và cao ≈280 px (= 348.92 − 69.24), bao trọn người đứng quay lưng.
3. Hai ngưỡng chọn là `0.20` và `0.60` (theo output ô so sánh ngưỡng trên ảnh `kitchen`):
   - `0.20`: **17** prediction (person ×2, bowl ×5, oven ×2, cup ×2, spoon ×3, potted plant, dining table, bottle)
   - `0.60`: **6** prediction (person ×2, bowl ×2, oven ×2)
4. Hạ ngưỡng thì giữ được nhiều vật thật hơn nhưng cũng sinh thêm hộp sai. Ví dụ, từ 0.60 xuống 0.20 số hộp tăng
   từ 6 lên 17, gần gấp 3. Trong 11 hộp thêm vào ở 0.20, `potted plant` thực ra là bó thảo mộc treo, tức một
   false positive. Reviewer phải xem thêm nhiều hộp điểm thấp: loại hộp nhận nhầm (false positive), gộp các hộp
   trùng trên cùng một vật, sửa lớp sai (ví dụ cup/bowl dễ lẫn nhau) và chỉnh lại biên hộp. Khối lượng QC tăng theo
   số hộp. Ngưỡng chỉ dùng để lọc prediction, không phải quy tắc bỏ vật khi tạo ground truth: chiếc `dining table`
   chỉ xuất hiện ở 0.20 nhưng vẫn là vật thật cần gán nhãn.
5. **Quy tắc box:** vẽ một hộp chữ nhật sát nhất bao trọn **phần nhìn thấy** của mỗi vật thể (mỗi vật một hộp). Cạnh hộp
   chạm điểm ngoài cùng của vật, không chừa nền thừa quá vài pixel. Không đoán và vẽ phần bị che hoặc nằm ngoài khung ảnh.
6. Vật bị cắt mép: `person` 0.61, bbox [0.12, 263.18, 61.55, 310.78]. Chỉ thấy một cánh tay thò vào từ mép trái
   (x≈0) của ảnh, phần còn lại của người nằm ngoài khung. Tôi sẽ **viết thành quy tắc guideline**: “vật bị cắt mép
   ảnh vẫn được gán nếu nhận ra được lớp; hộp chỉ bao phần nằm trong ảnh”. Lý do là tình huống này lặp lại ở rất nhiều
   ảnh và có cách xử lý rõ ràng. Chỉ escalation khi phần nhìn thấy quá nhỏ đến mức không chắc là lớp gì.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=`kitchen-001` ; `class_name`=`person` ; `score`=0.899 ; `polygon_point_count`=348 ;
   4 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0]]
2. Box chỉ cho 4 số (hình chữ nhật), nên bao cả nền giữa hai chân và hai bên người. Polygon 348 điểm bám theo đường viền
   thật: tóc, khuỷu tay chống hông, khe giữa hai chân, vạt tạp dề. Polygon cho biết **pixel nào thuộc vật**, nên tính
   được diện tích và hình dạng thật, đồng thời tách được vật khỏi nền và khỏi vật bên cạnh.
3. `instance_id` (ví dụ `kitchen-001`) là mã định danh **từng cá thể trong một ảnh**, để phân biệt hai vật cùng lớp.
   Ảnh `kitchen` có 4 `bowl` (`kitchen-002`, `-003`, `-006`, `-010`), cùng lớp nhưng là 4 instance khác nhau.
   - Không phải class ID: class ID (ví dụ 45 = bowl) cho biết **loại** vật và giống nhau cho mọi cái bát.
   - Không phải tracking ID: tracking ID theo dõi **cùng một vật qua nhiều khung hình video**. Còn `instance_id` chỉ
     có nghĩa trong một ảnh tĩnh, và `kitchen-001` không liên quan gì tới `traffic-001`.
4. **Quy tắc vẽ biên mask:** đường viền đi sát mép ngoài phần nhìn thấy của vật. Bỏ các phần bị vật khác che lên.
   Nếu vật có lỗ hở lộ nền (ví dụ khe giữa hai chân) thì không tô phần nền đó. Mỗi pixel chỉ thuộc **một** instance,
   các mask không chồng lên nhau.
5. Hai vật chạm/che nhau: mask `dining table` (`kitchen-005`, 598 điểm) và các `bowl`/hộp mứt đặt trên bàn. Trên ảnh,
   mask bàn màu xanh phủ gần như toàn bộ mặt bàn, kể cả những đồ vật đặt trên đó. Các `bowl` 0.53 và 0.74 ở góc
   trái dưới cũng chồng lên nhau. Tôi xử lý bằng **guideline**: “vật nằm trên che vật nằm dưới; mask bàn trừ đi phần
   bị bát/hộp che”. Lý do là đây là quy tắc chung cho mọi cảnh có đồ đặt trên bàn.
   Ngược lại, `potted plant` (`kitchen-004`, 0.63) thực ra là **bó thảo mộc treo khô**. Không rõ nó có thuộc
   “potted plant” hay không, và taxonomy COCO-80 không có lớp phù hợp. Ca này tôi **escalation** hỏi Lab Coach
   vì đó là quyết định về taxonomy, annotator không nên tự quyết.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **Một nhãn cho cả ảnh**: `class_id` + `class_name` + `taxonomy_name` | `kitchen` → `gong` 0.42 (sai, ImageNet không có lớp “kitchen”); `traffic` có taxi, xe buýt, xe tải nhưng chỉ được chọn 1 lớp | Xem cả ảnh, chọn một lớp theo quy tắc chủ thể chính; nếu không có lớp phù hợp thì escalation | Lớp có đúng taxonomy không, có theo đúng quy tắc chủ thể chính không, các annotator có chọn thống nhất không |
| Phát hiện vật thể | **Một hộp xyxy (pixel) + một lớp cho mỗi vật** | `person` 0.61 [0.12, 263.18, 61.55, 310.78] chỉ là cánh tay bị cắt mép; cup 0.45/0.38 và bowl nằm sát nhau dễ lẫn; nhiều nồi chảo treo không có hộp (COCO-80 không có lớp “pan”) | Vẽ hộp sát phần nhìn thấy cho mọi vật thuộc taxonomy, gán lớp, không bỏ sót vật nhỏ | Hộp sát chưa, có thiếu vật không, có hộp trùng không, lớp đúng không (cup hay bowl) |
| Instance segmentation | **Một polygon/mask + lớp + `instance_id` cho mỗi cá thể** | Mask `dining table` (`kitchen-005`) phủ cả đồ trên bàn; hai `bowl` chồng nhau; `potted plant` (`kitchen-004`) thực ra là thảo mộc treo | Tô biên sát từng cá thể, tách vật chạm nhau, mỗi pixel thuộc một instance | Biên có sát không, có rò sang vật khác hoặc nền không, có gộp nhầm hai instance thành một không, lớp có hợp lý không |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai mà notebook đã cố định và kiểm tra checksum. Không đưa ảnh cá nhân hoặc dữ liệu nội bộ
   vào lab, và không ghi họ tên, MSSV hay thông tin cá nhân vào báo cáo hoặc output.
2. Tôi dừng ngay, không xử lý tiếp, và báo cho **Lab Coach / GV phụ trách lớp** qua kênh hỗ trợ của lớp.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
