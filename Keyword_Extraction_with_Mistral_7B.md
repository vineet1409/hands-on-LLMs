# **Introducing `KeyLLM`: Keyword Extraction with Mistral 7B**
*Combining `KeyBERT` with Large Language Models*
<br>
<div>

<img src="https://github.com/MaartenGr/KeyBERT/assets/25746895/7351a3f1-f3a4-4911-8535-664f37adec78" width="750"/>
</div>



---
        
💡 **NOTE**: We will want to use a GPU to run both Llama2 as well as KeyBERT for this use case. In Google Colab, go to
**Runtime > Change runtime type > Hardware accelerator > GPU > GPU type > T4**.

---

We will start by installing a number of packages that we are going to use throughout this example:


```python
%%capture
!pip install --upgrade git+https://github.com/UKPLab/sentence-transformers
!pip install keybert ctransformers[cuda]
!pip install --upgrade git+https://github.com/huggingface/transformers

```


```python
!huggingface-cli login
```

    
        _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
        _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
        _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
        _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
        _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|
    
        To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
    Token: 
    Add token as git credential? (Y/n) n
    Token is valid (permission: write).
    Your token has been saved to /root/.cache/huggingface/token
    Login successful



🔥 **TIP** 🔥: We are instaling `sentence-transformers` from its main branch since it has a fix for community detection which we will using in the last few use cases. We do the same for `transformers` since it does not yet support the Mistral architecture.

# <img src="https://github.com/MaartenGr/KeyBERT/assets/25746895/5bb064ee-7545-48a5-8248-6f4afb8cfd9d" width="30"> **Loading the Model**

In previous tutorials, we demonstrated how we could quantize the original model's weight to make it run without running into memory problems.

Over the course of the last few months, [TheBloke](https://huggingface.co/TheBloke) has been working hard on doing the quantization for hundreds of models for us.

This way, we can download the model directly which will speed things up quite a bit.

We'll start with loading the model itself. We will ofload 50 layers to the GPU. This will reduce RAM usage and use VRAM instead. If you are running into memory errors, reducing this parameter (`gpu_layers`) might help!


```python
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    model_type="mistral",
    gpu_layers=50,
    hf=True
)
```


    Fetching 1 files:   0%|          | 0/1 [00:00<?, ?it/s]



    config.json:   0%|          | 0.00/31.0 [00:00<?, ?B/s]



    Fetching 1 files:   0%|          | 0/1 [00:00<?, ?it/s]



    mistral-7b-instruct-v0.1.Q4_K_M.gguf:   0%|          | 0.00/4.37G [00:00<?, ?B/s]


    The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.



    0it [00:00, ?it/s]


After having loaded the model itself, we want to create a 🤗 Transformers pipeline.

The main benefit of doing so is that these pipelines are found in many tutorials and are often used in packages as backend. Thus far, `ctransformers` is not yet natively supported as much as `transformers`.

Loading the Mistral tokenizer with `ctransformers` is not yet possible as the model is quite new. Instead, we use the tokenizer from the original repository instead.


```python
from transformers import AutoTokenizer, pipeline

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Pipeline
generator = pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    max_new_tokens=50,
    repetition_penalty=1.1
)
```


    tokenizer_config.json:   0%|          | 0.00/1.47k [00:00<?, ?B/s]



    tokenizer.model:   0%|          | 0.00/493k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.80M [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/72.0 [00:00<?, ?B/s]


# 📄 **Prompt Engineering**


Let's see if this works with a very basic example:


```python
response = generator("What is 1+1?")
print(response[0]["generated_text"])
```

    What is 1+1?
    A: 2


Perfect! It can handle a very basic question. For the purpose of keyword extraction, let's explore whether it can handle a bit more complexity.


```python
prompt = """
I have the following document:
* The website mentions that it only takes a couple of days to deliver but I still have not received mine

Extract 5 keywords from that document.
"""
response = generator(prompt)
print(response[0]["generated_text"])
```

    
    I have the following document:
    * The website mentions that it only takes a couple of days to deliver but I still have not received mine
    
    Extract 5 keywords from that document.
    
    **Answer:**
    1. Website
    2. Mentions
    3. Deliver
    4. Couple
    5. Days


It does great! However, if we want the structure of the output to stay consistent regardless of the input text we will have to give the LLM an example.

This is where more advanced prompt engineering comes in. As with most Large Language Models, Mistral 7B expects a certain prompt format. This is tremendously helpful when we want to show it what a "correct" interaction looks like.

The prompt template is as follows:

<br>
<div>
<img src="https://github.com/MaartenGr/KeyBERT/assets/25746895/aba167b1-93e6-44ab-a39b-4aab85c858c0" width="850"/>
</div>


Based on that template, let's create a template of our for keyword extraction.

It needs to have two components:
* `Example prompt` - This will be used to show the LLM what a "good" output looks like
* `Keyword prompt` - This will be used to ask the LLM to extract the keywords

The first component, the `example_prompt`, will simply be an example of correctly extracting the keywords in the format that we are interested.

Especially the **format** is a key component since it will make sure that the LLM will always output keywords the way we want:


```python
example_prompt = """
<s>[INST]
I have the following document:
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST] meat, beef, eat, eating, emissions, steak, food, health, processed, chicken</s>"""
```

The second component, the `keyword_prompt`, will essentially be a repeat of the `example_prompt` but with two changes:
* It will not have an output yet. That will be generated by the LLM.
* We use of `KeyBERT`'s **[DOCUMENT]** tag for indicating where the input document will go.

We can use the **[DOCUMENT]** to insert a document at a location of your choice. Having this option helps us to change the structure of the prompt if needed without being set on having the prompt at a specific location.


```python
keyword_prompt = """
[INST]

I have the following document:
- [DOCUMENT]

Please give me the keywords that are present in this document and separate them with commas.
Make sure you to only return the keywords and say nothing else. For example, don't say:
"Here are the keywords present in the document"
[/INST]
"""
```

Lastly, we combine the two prompts to create our final template:


```python
prompt = example_prompt + keyword_prompt
print(prompt)
```

    
    <s>[INST]
    I have the following document:
    - The website mentions that it only takes a couple of days to deliver but I still have not received mine.
    
    Please give me the keywords that are present in this document and separate them with commas.
    Make sure you to only return the keywords and say nothing else. For example, don't say:
    "Here are the keywords present in the document"
    [/INST] meat, beef, eat, eating, emissions, steak, food, health, processed, chicken</s>
    [INST]
    
    I have the following document:
    - [DOCUMENT]
    
    Please give me the keywords that are present in this document and separate them with commas.
    Make sure you to only return the keywords and say nothing else. For example, don't say:
    "Here are the keywords present in the document"
    [/INST]
    


Now that we have our final prompt template, we can start exploring a couple of interesting new features in `KeyBERT` with `KeyLLM`. We will start with exploring `KeyLLM` only using Mistral's 7B model

# 🗝️ Keyword Extraction with `KeyLLM`

Keyword extraction with vanilla `KeyLLM` couldn't be more straightforward; we simply ask it to extract keywords from a document.

<br>
<div>
<img src="https://github.com/MaartenGr/BERTopic/assets/25746895/bdcd7ae5-41d6-4687-828a-e7b245cef593" width="650"/>
</div>



This idea of extracting keywords from documents through an LLM is straightforward and allows for easily testing your LLM and its capabilities.

Using `KeyLLM` is straightforward, we start by loading our LLM throught `keybert.llm.TextGeneration` and give it the prompt template that we created before.

🔥 **TIP** 🔥: If you want to use a different LLM, like ChatGPT, you can find a full overview of implemented algorithms [here:](https://maartengr.github.io/KeyBERT/guides/llms.html)


```python
from keybert.llm import TextGeneration
from keybert import KeyLLM

# Load it in KeyLLM
llm = TextGeneration(generator, prompt=prompt)
kw_model = KeyLLM(llm)
```

After preparing our `KeyLLM` instance, it is as simple as running `.extract_keywords` over your documents:


```python
documents = [
"The website mentions that it only takes a couple of days to deliver but I still have not received mine.",
"I received my package!",
"Whereas the most powerful LLMs have generally been accessible only through limited APIs (if at all), Meta released LLaMA's model weights to the research community under a noncommercial license."
]

keywords = kw_model.extract_keywords(documents); keywords
```




    [['website',
      'mention',
      'days',
      'deliver',
      'receive',
      'coupler',
      'still',
      'have',
      'not',
      'received',
      'mine.'],
     ['package',
      'received',
      'delivery',
      'shipment',
      'order',
      'product',
      'item',
      'box',
      'mail',
      'courier'],
     ['LLM',
      'API',
      'accessibility',
      'release',
      'license',
      'research',
      'community',
      'model',
      'weights',
      'Meta',
      'power',
      'availability',
      'commercial',
      'noncommercial',
      'language',
      'models',
      'development',
      'collaboration',
      'innovation',
      'openness',
      'sharing',
      'knowledge',
      'resources']]



These seem like a great set of keywords!

You can play around with the prompt to specify the kind of keywords you want extracted, how long they can be, and even in which language they should be returned if your LLM is multi-lingual.

# 🚀 Efficient Keyword Extraction with `KeyLLM`

Iterating your LLM over thousands of documents is not the most efficient approach! Instead, we can leverage embedding models to make the keyword extraction a bit more efficient.

This works as follows. First, we embed all of our documents and convert them to numerical representations. Second, we find out which documents are most similar to one another. We assume that documents that are highly similar will have the same keywords, so there would be no need to extract keywords for all documents. Third, we only extract keywords from 1 document in each cluster and assign the keywords to all documents in the same cluster.

This is much more efficient and also quite flexibel. The clusters are generated purely based on the similarity between documents, without taking cluster structures into account. In other words, it is essentially finding near-duplicate documents that we expect to have the same set of keywords.

<br>
<div>
<img src="https://github.com/MaartenGr/BERTopic/assets/25746895/d7e13668-590a-424c-9ef8-6dc886a19597" width="650"/>
</div>



To do this with `KeyLLM`, we embed our documents beforehand and pass them to `.extract_keywords`. The threshold indicates how similar documents will minimally need to be in order to be assigned to the same cluster.

Increasing this value to something like .95 will identify near-identical documents whereas setting it to something like .5 will identify documents about the same topic.


```python
from keybert import KeyLLM
from sentence_transformers import SentenceTransformer

# Extract embeddings
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embeddings = model.encode(documents, convert_to_tensor=True)
```


    modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]



    config_sentence_transformers.json:   0%|          | 0.00/124 [00:00<?, ?B/s]



    README.md:   0%|          | 0.00/90.3k [00:00<?, ?B/s]



    sentence_bert_config.json:   0%|          | 0.00/52.0 [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/743 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/133M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/366 [00:00<?, ?B/s]



    vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/711k [00:00<?, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/125 [00:00<?, ?B/s]



    1_Pooling/config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]



```python
# Load it in KeyLLM
kw_model = KeyLLM(llm)

# Extract keywords
keywords = kw_model.extract_keywords(documents, embeddings=embeddings, threshold=.5)
```


```python
keywords
```




    [['website',
      'mention',
      'days',
      'deliver',
      'receive',
      'coupler',
      'still',
      'have',
      'not',
      'received',
      'mine.'],
     ['website',
      'mention',
      'days',
      'deliver',
      'receive',
      'coupler',
      'still',
      'have',
      'not',
      'received',
      'mine.'],
     ['LLM',
      'API',
      'accessibility',
      'release',
      'license',
      'research',
      'community',
      'model',
      'weights',
      'Meta',
      'power',
      'availability',
      'commercial',
      'noncommercial',
      'language',
      'models',
      'development',
      'collaboration',
      'innovation',
      'openness',
      'sharing',
      'knowledge',
      'resources']]



In this example, we can see that the first two documents were clustered together and received the same keywords. Instead of passing all three documents to the LLM, we only pass two documents. This can speed things up significantly if you have thousands of documents.

# 🏆 Efficient Keyword Extraction with `KeyBERT` & `KeyLLM`

Before, we manually passed the embeddings to `KeyLLM` to essentially do a zero-shot extraction of keywords. We can further extend this example by leveraging `KeyBERT`.

Since `KeyBERT` generates keywords and embeds the documents, we can leverage that to not only simplify the pipeline but suggest a number of keywords to the LLM.

These suggested keywords can help the LLM decide on the keywords to use. Moreover, it allows for everything within `KeyBERT` to be used with `KeyLLM`!


<br>
<div>
<img src="https://github.com/MaartenGr/BERTopic/assets/25746895/01b4b831-7dd3-4ea9-be81-6dff4cc9a32b" width="450"/>
</div>


This efficient keyword extraction with both `KeyBERT` and `KeyLLM` only requires three lines of code! We create a KeyBERT model and assign it the LLM with the embedding model we previously created:


```python
from keybert import KeyLLM, KeyBERT

# Load it in KeyLLM
kw_model = KeyBERT(llm=llm, model='BAAI/bge-small-en-v1.5')

# Extract keywords
keywords = kw_model.extract_keywords(documents, threshold=.5)
```


```python
keywords
```




    [['website',
      'mention',
      'days',
      'deliver',
      'receive',
      'coupler',
      'still',
      'have',
      'not',
      'received',
      'mine.'],
     ['website',
      'mention',
      'days',
      'deliver',
      'receive',
      'coupler',
      'still',
      'have',
      'not',
      'received',
      'mine.'],
     ['LLM',
      'API',
      'accessibility',
      'release',
      'license',
      'research',
      'community',
      'model',
      'weights',
      'Meta',
      'power',
      'availability',
      'commercial',
      'noncommercial',
      'language',
      'models',
      'development',
      'collaboration',
      'innovation',
      'openness',
      'sharing',
      'knowledge',
      'resources']]



And that is it! With `KeyLLM` you are able to use Large Language Models to help create better keywords. We can choose to extract keywords from the text itself or ask the LLM to come up with keywords.

By combining `KeyLLM` with `KeyBERT`, we increase its potential by doing some computation and suggestions beforehand.


🔥 **TIP** 🔥: You can use `[CANDIDATES]` to pass the generated keywords in KeyBERT to the LLM as candidate keywords. That way, you can tell the LLM that KeyBERT has already generated a number of keywords and ask it to improve them.


```python
!jupyter nbconvert --to markdown Keyword_Extraction_with_Mistral_7B.ipynb
```

    [NbConvertApp] WARNING | pattern 'Keyword_Extraction_with_Mistral_7B.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        overwrite base name use for output files.
                    can only be used when converting one notebook at a time.
        Default: ''
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    



```python

```
