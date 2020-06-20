#python generate_dataset.py
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../../data/Instagram/caption_dataset/train.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../../data/Instagram/caption_dataset/test1.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../../data/Instagram/caption_dataset/test2.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../../data/Instagram/hashtag_dataset/train.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../../data/Instagram/hashtag_dataset/test1.txt
python cnn_feature_extractor.py --gpu_id 0 --batch_size 32 --input_fname ../../data/Instagram/hashtag_dataset/test2.txt
