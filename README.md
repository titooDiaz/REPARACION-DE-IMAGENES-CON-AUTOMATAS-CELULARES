<img width="1584" height="672" alt="Project image" src="https://github.com/user-attachments/assets/29c097ec-36c1-470e-a601-46cdb8032806" />

## How to Use

1. **Download the dataset**
   We use [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset, but you can use any similar dataset

2. **Prepare the data folder**
   In the root directory, create a folder called `data`, inside `data`, create another folder called `img_align_celeba`. Put all your images inside `img_align_celeba`

3. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create models folder**

   * Inside the `model` folder, create a new folder called `models`

5. **Train the model**
   Run:

   ```bash
   python train.py
   ```

   After training, a file called `nca.pth` will appear in the `models` folder.

6. **Test the model**

   * Add a test image inside the `model` folder
   * Name it `test.jpeg`

7. **Visualize results**
   Run:

   ```bash
   python visualize.py
   ```

   The result will be saved as `reconstruction.mp4`.
