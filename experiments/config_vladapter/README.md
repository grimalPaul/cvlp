# Help

All possible argument for config for a model
Some arguments are not compatible with each others

please add foleder `snap/`

```json
{      
    ///////train args//////////
    // Data Splits
    "train":"train",
    "valid":"valid",
    "test":null,
    "test_only":false,

    "submit":true,

    // Quick experiments
    "train_topk":-1,
    "valid_topk":-1,


    // CPU/GPU
    "multiGPU":false,
    "fp16":false,
    "distributed":false,
    "num_workers":0,
    "local_rank":-1,
    // Checkpoint
    "output": "snap/test",
    "load":"#TODO",
    "from_scratch" : false,
    "run_name":"",

    // Training
    "batch_size":256,
    "valid_batch_size":null,
    "optim":"adamw",
    "warmup_ratio":0.05,
    "weight_decay":0.01,
    "clip_grad_norm":-1.0,
    "gradient_accumulation_steps":1,
    "lr":1e-4,
    "vis_lr":1e-4,
    "vis_weight_decay":0.01,,
    "adam_eps":1e-6,
    "adam_beta1":0.9,
    "adam_beta2":0.999,
    "epochs":12,
    "dropout":0.1,    
    "losses":"lm,obj,attr,feat", 
    "log_train_accuracy":false,
    // compute results before any training
    "dry":false,
    //ajouter une facon de faire les commentaires
    "comment":"",
    
    
    
   
    ///////////// DATASET //////////////////////
    
   
    // add specific prompt for each task
    "use_tasks_prompt":true,

    "n_ground":1, // vcr
    "word_mask_rate": 0.15,
    "obj_mask_rate":0.15,

    "caption_only": true,
    "coco_only": true,
    "caption_cocoonly":true,

    "do_lower_case":true,
    "oscar_tags":true,

    "prefix":"span, denoise, ....",

    // if self.args.prefix is None:
    // prefix = f'{self.args.prompt}'
    "prompt" :"vqa: ",
    "post_prompt":"",

    //type of input features
    // "RN50", "RN101", "RN50x4", "ViT", "butd", "raw_RN50", "raw_RN101", "raw_RN50x4", "raw_ViT
    "feature_type":"butd" 



    /////////// Model Config ////////////
    "backbone":"t5-base",
    "tokenizer": null,

    "feat_dim":2048,
    "pos_dim":4,
    "image_size":"448,448",

    "use_vision":true,
    "use_vis_order_embedding":true,
    "use_vis_layer_norm":true,
    "individual_vis_layer_norm":true,
    "share_vis_lang_layer_norm":true,

    "n_boxes":36,
    "max_n_boxes":36,
    
    "additional_visual_embedding_layers":0,

    // OneDDownsample xor Downsample xor SparseSample, to reduce size of visual input or embed
    "sparse_sample":true,
    "downsample":true,
    "oneddownsample":true,
    "expand_vis_embedding": true,
    "n_image_tokens" :4, // pas utilisé avec VLT5
    "vis_use_transformer": true,


    "encoder_prompt_len":0,
    "decoder_prompt_len":0,
    // multiple prompt = one prompt per task
    // single prompt = one prompt for all tasks
    "use_single_prompt":true or,
    "unfreeze_language_model":false,
    "unfreeze_layer_norms":false,
    "use_attn_prefix":false,
    "mid_dim":768,
    

    //choose one type of adapter
    "use_adapter": false,
    "use_hyperformer":false,
    "use_compacter":false,
    "use_lradapter":false,

    "use_single_adapter":false,
    
    //hyperperformer
    "efficient_unique_hyper_net":false,
    "unique_hyper_net":false,
    // projected_task_embedding_dim for hyperformer, -1 means using the default value in the config
    "projected_task_embedding_dim": -1 ,
    
    // to train or not the vision model
    // in our case we dont touch the vision model
    "unfreeze_vis_encoder":false,
    "unfreeze_vis_last_layer":false,
    "unfreeze_batch_norms":false,

    // share Downsample and upsample between adapter
    "share_down_sampler":true,
    "share_up_sampler":true,
    
    // Compacter
    "hypercomplex_division":4,
    "phm_rank":1,
    "shared_phm_rule":true,
    "factorized_phm":true,
    "phm_init_range": 0.01,
    "shared_phm_rule_over_tasks":true,

    // for hyperformer, adapter, compacter, lraAdapter
    "add_adapter_cross_attn":true,
    
    // low-rank adapter, in which each adapter is composed of two rank-one matrices.
    "low_rank_rank" : 1,
    
    // config for the visual model
    // we wont train vis part
    "vis_pooling_output":false,
    "use_vis_adapter":false,
    "use_separate_optimizer_for_visual":false, "use_adam_for_visual":true, //SGD if false
    
    // partial eval
    // T5LayerNorm, nn.LayerNorm in eval()
    "freeze_ln_statistics":false,
    //nn.BatchNorm2d in eval()
    "freeze_bn_statistics": false,

    "add_layer_norm_before_adapter": true,
    "add_layer_norm_after_adapter":true,

    //adapter for vision
    "vis_adapter_type":"middle-bottleneck",
    "vis_reduction_factor": 2,
    "reduction_factor":16,

    //apply some transformation on image
    "use_data_augmentation":true,
    //only used to full pretrain the model
    "deepspeed":null,
    
    // load vis encoder with or without Batchnorm 
    "remove_bn_vis_adapter": true,
    //unfreeze lm head
    //output layer
    "unfreeze_lm_head": true,
    //output adapter ?
    "use_lm_head_adapter":true,

    // use lora
    "use_lora":false,
    "lora_dim":4,
    "lora_alpha":32,
    "use_single_lora": true,

    // Inference
    "num_beams" :1,
    "gen_max_length" :20,


    //// dataset, config model, training
    //multitask
    "multitask_sampling":"roundrobin",
    "tasks": "",
    "testing":true,
    // une valeur dans les adapteurs qui peut être suivi
    "lambda_z":0.001,
    "track_z":true,


}
```

```bash
 # Pretraining
    parser.add_argument('--ground_upsample', type=int, default=1)
    parser.add_argument('--ground_weight', type=int, default=1)
    parser.add_argument('--itm_cocoonly', default=True, type=str2bool)
    parser.add_argument('--single_vqa_prefix', action='store_true')

    # COCO Caption
    parser.add_argument('--no_prefix', action='store_true')

    # VQA
    parser.add_argument("--raw_label", action='store_true')
    parser.add_argument("--answer_normalize", action='store_true')
    parser.add_argument("--classifier", action='store_true')
    parser.add_argument("--test_answerable", action='store_true')

    # RefCOCOg
    parser.add_argument('--RefCOCO_GT', action='store_true')
    parser.add_argument('--RefCOCO_BUTD', action='store_true')
    parser.add_argument("--shuffle_boxes", action='store_true')
    parser.add_argument('--vis_pointer', action='store_true')

    # Classification
    parser.add_argument('--cls_task', type=str, default='tinyimagenet')
```
