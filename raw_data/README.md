# Raw Dataset Files Folder

The folder contains raw dataset files, which consists of 5 files as below:

- product_train.csv, information of the items.
- sessions_train.csv, interaction file for training.
- sessions_test_task1.csv, test file for task 1.
- sessions_test_task2.csv, test file for task 2.
- sessions_test_task3.csv, test file for task 3.

For more information about the three tasks and the datasets, please visit [Amazon KDD Cup 2023 Website](https://www.aicrowd.com/challenges/amazon-kdd-cup-23-multilingual-recommendation-challenge).

Attention: The download tool in China is not friendly, which would have a low download speed. So we share a copy of the raw data in AutoDL-A100 machine and in [Rec Web Drive](https://rec.ustc.edu.cn/).
- The copy in AutoDL-A100: `/root/autodl-tmp/xiaolong/WorkSpace/KDD_CUP_2023/kdd_cup_2023_data/`. You can copy all the files into your own `raw_data` folder or create a soft link as below:

    ```bash
    ln -s /root/autodl-tmp/xiaolong/WorkSpace/KDD_CUP_2023/kdd_cup_2023_data/* {your folder}/raw_data/
    ```

- The link of Rec Web Drive is `https://rec.ustc.edu.cn/share/415c7910-c635-11ed-8237-b3365e7fc847`. However, it's not the direct download link. You should open the link in browser in your computer first and then click button to get the real download link (do not download but copy the download link).