import os

root_path = os.path.dirname(__file__)

configs = {
            "config": 
                    os.path.join(root_path, "config/bert-base-chinese-config.json"),
            "vocab": 
                    os.path.join(root_path, "config/bert-base-chinese-vocab.txt"),
            "ner_model": 
                    os.path.join(root_path, "saved_models/ner.bin"),
            "candiate_entity_similarity_model": 
                    os.path.join(root_path, "saved_models/candiate_entity_similarity.bin"),
            "path_similarity_mode": 
                    os.path.join(root_path, "saved_models/path_similarity.bin"),
            "lgb_model_link": 
                    os.path.join(root_path, "saved_models/lgb_model_link.txt"),
            "lgb_model_path": 
                    os.path.join(root_path, "saved_models/lgb_model_path_rank.txt")
            }

print(os.path.dirname(__file__))