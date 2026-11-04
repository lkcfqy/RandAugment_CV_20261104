# AeBAD-S official subset

Source: https://github.com/zhangzilongc/MMR (commit 22d1b171687509024e6f5b53b5a5e772d653d32b).
Original publication: Zilong Zhang, Zhibin Zhao, Xingwu Zhang, Chuang Sun, Xuefeng Chen, “Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction”, https://arxiv.org/abs/2304.02216v2.

Dataset license: CC BY 4.0; the original LICENSE-DATASET is retained alongside this file.
Official README download: https://drive.google.com/file/d/14wkZAFFeudlg0NMFLsiGwS0E593b-lNo/view .

Only AeBAD_S train/good and test/ground_truth same + illumination were extracted, using public HTTP Range requests from the original AeBAD.zip. Original image and mask bytes and names are unchanged. macOS dotfiles, resource forks and unrequested domains/video data are not extracted. 521 train images, 962 test images, 657 anomaly masks. The train set contains 520 distinct raw/decoded images because two published paths contain the same image; no file was removed.

The .png-named image files are actually JPEG-coded RGB; masks are PNG. Use byte-aware image decoding. No entity/batch mapping is supplied; filenames must not be treated as independent specimens.
