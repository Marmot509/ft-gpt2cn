import logging
import os
import sys
import torch
import json
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    Trainer,
    set_seed,
)

from arguments import ModelArguments, DataTrainingArguments

from preprocess import sanity_check, MultiTurnDataset, InputOutputDataset

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.ddp_find_unused_parameters = False
    training_args.save_safetensors = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)
    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.float()

    with open(data_args.train_file, "r", encoding="utf-8") as f:
        if data_args.train_file.endswith(".json"):
            train_data = json.load(f)
        elif data_args.train_file.endswith(".jsonl"):
            train_data = [json.loads(line) for line in f]

    if data_args.train_format == "multi-turn":
        train_dataset = MultiTurnDataset(
            train_data,
            tokenizer,
            data_args.max_seq_length,
        )
    elif data_args.train_format == "input-output":
        train_dataset = InputOutputDataset(
            train_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
    else:
        raise ValueError(f"Unknown train format: {data_args.train_format}")
    if training_args.local_rank < 1:
        sanity_check(train_dataset[0]['input_ids'], train_dataset[0]['labels'], tokenizer)

    ### Add validation data loading by Xin
    with open(data_args.val_file, "r", encoding="utf-8") as f:
        if data_args.val_file.endswith(".json"):
            val_data = json.load(f)
        elif data_args.val_file.endswith(".jsonl"):
            val_data = [json.loads(line) for line in f]

    if data_args.train_format == "input-output":
        val_dataset = InputOutputDataset(
            val_data,
            tokenizer,
            data_args.max_source_length,
            data_args.max_target_length,
        )
    else:
        raise ValueError(f"Unknown validation format: {data_args.train_format}")
    print(f"Validation dataset size: {len(val_dataset)}")


    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.save_state()


if __name__ == "__main__":
    main()
