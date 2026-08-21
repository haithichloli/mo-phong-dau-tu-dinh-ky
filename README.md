# Mô phỏng đầu tư định kỳ

Ứng dụng Streamlit mô phỏng việc góp vốn hàng tháng vào nhiều kênh đầu tư.
Các kênh thị trường sử dụng lợi suất tháng lịch sử tải từ Yahoo Finance và
phương pháp lấy mẫu ngẫu nhiên có hoàn lại để tạo nhiều kịch bản tương lai.

## Chức năng

- Nhập vốn ban đầu và số tiền đầu tư mỗi tháng.
- Chọn thời gian đầu tư và số kịch bản mô phỏng.
- So sánh tiết kiệm, ETF VN30, S&P 500, vàng và Bitcoin.
- Hiển thị kịch bản thấp 5%, trung vị và kịch bản cao 95%.
- Tính xác suất giá trị cuối kỳ thấp hơn tổng vốn đã góp.
- Biểu đồ dải kịch bản, so sánh kênh và phân phối giá trị cuối kỳ.
- Tải dữ liệu mới và lưu cache trong 6 giờ.
- Tự dùng bản dữ liệu lịch sử dự phòng khi Yahoo Finance tạm thời không phản hồi.

## Chạy cục bộ

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Lưu ý

Ứng dụng chỉ phục vụ học tập và mô phỏng. Đây không phải dự báo hoặc khuyến
nghị đầu tư. Mô hình chưa tính thuế, phí giao dịch, trượt giá, lạm phát hay
tác động tỷ giá đối với tài sản nước ngoài.

