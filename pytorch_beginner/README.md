# Pytorch tutorial

<p align="center">
<img width="400" alt="eval" src="images/comp-graph.png">
</p>

➡ Muốn khởi tạo model trên pytorch thì cần phải khởi tạo thông qua một class kế thừa module nn.Module. (Khởi tạo model trong TF2 thì cũng phải viết layer kế thừa qua tf.keras.layers.Layer)
➡ Phải khai báo các layer trong hàm __init__() của class.
➡ Xác định output thông qua hàm forward() của class (Class based thì cũng phải xác định output thông qua hàm call).
➡ Để truyền được model vào huấn luyện thì phải wrap data vào DataLoader (Tf có tf.data optimize nhanh hơn).
➡ Qúa trình huấn luyện thì phải trải qua tuần tự các bước: Khởi tạo gradient, feed forward, tính loss function, backward.
➡ Luôn phải khai báo device cho các object khi chuyển đổi giữa CPU và GPU.
➡ Tuy nhiều bước hơn nhưng pytorch cho phép mình can thiệp sâu hơn vào mô hình.