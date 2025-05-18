# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")
# # model = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")

# tokenizer.save_pretrained("./biomed_tokenizer")
# # model.save_pretrained("./biomed_model")


from transformers import AutoTokenizer, AutoModelForCausalLM

# model = AutoModelForCausalLM.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("ContactDoctor/Bio-Medical-Llama-3-8B")

# model.save_pretrained("./BioMedLlama-3-8B")
tokenizer.save_pretrained("./BioMedLlama-3-8B_tokenizer")