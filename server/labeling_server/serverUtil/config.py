class Config():

    train_img_list = './dataUtil/model_dataset/model_learn_images/train/*.jpg'
    valid_img_list = './dataUtil/model_dataset/model_learn_images/valid/*.jpg'

    train_txt = './dataUtil/model_dataset/train.txt'
    valid_txt = './dataUtil/model_dataset/valid.txt'

    input_size = 640
    batch_size = 4
    epochs = 2
    
    data_yaml = './dataUtil/model_dataset/data.yaml'
    weight_pt = './dataUtil/model_dataset/yolov5s.pt'

    # train_name = './dataUtil/model_weight_result'
    train_name = 'result'
    model_config = './dataUtil/model_dataset/yolov5s.yaml'