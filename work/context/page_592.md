# Generation bundle for page_592  (PDF page 592, printed page 568)

Page kind: content
Chapter: 19 The future of AI
Section: 19.1 The limitations of deep learning
Subsection: 19.1.2 Deep learning models are highly sensitive to phrasing and other distractors
Rendered page image (LOOK AT IT for formulas, figures, tables, layout): /Users/alireza/Coding/book_rag/book_companion/work/page_images/page_592.png
Output file to write: /Users/alireza/Coding/book_rag/book_companion/data/pages/page_592.json

## Extracted elements on this page (reading order; ids are used in `paragraphs[].ref`)

[p592_e03] (body) names in a text paragraph or variable names in a block of code, can significantly degrade LLM performance. Consider the well-known Alice in Wonderland riddle:1
[p592_e04] (quote) “Alice has N brothers and she also has M sisters. How many sisters does Alice’s brother have?”
[p592_e05] (body) The answer, of course, is M + 1 (Alice’s sisters plus Alice herself). For an LLM, asking the question with values commonly found in online instances of the riddle (like N = 3 and M = 2) will generally result in the correct answer, but try tweaking the values of M and N, and you will quickly get incorrect answers.
[p592_e06] (body) This oversensitivity to phrasing has given rise to the concept of prompt engineering. Prompt engineering is the art of formulating LLM prompts in a way that maximizes performance on a task. For instance, it turns out that adding the instruction “Please think step by step” to a prompt that involves reasoning can significantly boost performance. The term prompt engineering is a very optimistic framing of the underlying issue: “Your models are better than you know! You just need to use them right!” A more negative framing would be to point out that for any query that seems to work, there’s a range of minor changes that have the potential to tank performance. To what extent do LLMs understand something if you can break their understanding with simple rewordings? What’s behind this phenomenon is that an LLM is a big parametric curve—a medium for storing knowledge and programs where you can interpolate between any two objects to produce infinitely many intermediate objects. Your prompt is a way to address a particular location of the database: if you ask, “How do you sort a list in Python? Answer like a pirate,” that’s a kind of database lookup, where you first retrieve a piece of knowledge (how to sort a list in Python) and then retrieve and execute a style transfer program (“Answer like a pirate”). Since the knowledge and programs indexed by the LLM are interpolative, you can move around in latent space to explore nearby locations. A slightly different prompt, like “Explain Python list sorting, but answer like a buccaneer” would still have pointed to a very similar location in the database, resulting in an answer that would be pretty close but not quite identical. There are thousands of variations you could have used, each resulting in a similar yet slightly different answer. And that’s why prompt engineering is needed. There is no a priori reason for your first, naive prompt to be optimal for your task. The LLM is not going to understand what you meant and then perform it in the best possible way—it’s merely going to fetch the program that your prompt points to, among many possible locations you could have landed on. Prompt engineering is the process of searching through latent space to find the lookup query that seems to perform best on your target task by trial and error. It’s no different from trying different keywords when doing a Google search. If LLMs actually understood what you asked them, there would be no need for this search process, since the amount of information conveyed about your target task does not change whether
[p592_e07] (footnote) 1 See also Marianna Nezhurina, Lucia Cipolina-Kun, Mehdi Cherti, and Jenia Jitsev, “Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models,” arXiv, https://arxiv .org/abs/2406.02061.

## Equation candidates (formulas written inline in the text)
- [p592_e05] For an LLM, asking the question with values commonly found in online instances of the riddle (like N = 3 and M = 2) will generally result in the correct answer, but try tweaking the values of M and N, and you will quickly get incorrect answers.

## Chapter outline
Chapter: 19 The future of AI (PDF pages 588-618)
  - 19.1 The limitations of deep learning (p.588)
      - 19.1.1 Deep learning models struggle to adapt to novelty (p.589)
      - 19.1.2 Deep learning models are highly sensitive to phrasing and other distractors (p.591)
      - 19.1.3 Deep learning models struggle to learn generalizable programs (p.593)
      - 19.1.4 The risk of anthropomorphizing machine-learning models (p.593)
  - 19.2 Scale isn’t all you need (p.594)
      - 19.2.1 Automatons vs. intelligent agents (p.595)
      - 19.2.2 Local generalization vs. extreme generalization (p.597)
      - 19.2.3 The purpose of intelligence (p.599)
      - 19.2.4 Climbing the spectrum of generalization (p.599)
  - 19.3 How to build intelligence (p.600)
      - 19.3.1 The kaleidoscope hypothesis (p.601)
      - 19.3.2 The essence of intelligence: Abstraction acquisition and recombination (p.602)
      - 19.3.3 The importance of setting the right target (p.602)
      - 19.3.4 A new target: On-the-fly adaptation (p.604)
      - 19.3.5 ARC Prize (p.605)
      - 19.3.6 The test-time adaptation era (p.606)
      - 19.3.7 ARC-AGI 2 (p.607)
  - 19.4 The missing ingredients: Search and symbols (p.608)
      - 19.4.1 The two poles of abstraction (p.609)
      - 19.4.2 Cognition as a combination of both kinds of abstraction (p.611)
      - 19.4.3 Why deep learning isn’t a complete answer to abstraction generation (p.612)
      - 19.4.4 An alternative approach to AI: Program synthesis (p.613)
      - 19.4.5 Blending deep learning and program synthesis (p.614)
      - 19.4.6 Modular component recombination and lifelong learning (p.616)
      - 19.4.7 The long-term vision (p.618)

## Previous page (PDF 591) text, for continuity
without addressing the more general underlying issue. Even already patched prompts will still fail if you make small changes to them!

19.1.2 Deep learning models are highly sensitive to phrasing and other distractors

A closely related problem is the extreme sensitivity of deep learning models to how their input is presented. For instance, image models are affected by adversarial examples, which are samples fed to a deep learning network that are designed to trick the model into misclassifying them. You’re already aware that it’s possible to do gradient ascent in input space to generate inputs that maximize the activation of some ConvNet filter—this is the basis of the filter visualization technique introduced in chapter 10.

Similarly, through gradient ascent, you can slightly modify an image to maximize the class prediction for a given class. By taking a picture of a panda and adding to it a gibbon gradient, we can get a neural network to classify the panda as a gibbon (see figure 19.3). This evidences both the brittleness of these models and the deep difference between their input-to-output mapping and our human perception.

Figure 19.3 An adversarial example: imperceptible changes in an image can upend a model’s classification of the image.

Similarly, LLMs suffer from an extremely high sensitivity to minor details in their prompts. Innocuous prompt modifications, such as changing place and people’s

## Next page (PDF 593) text, for continuity
your prompt uses the word “rewrite” instead of “rephrase” or whether you prefix your prompt with “Think step by step.” Never assume that the LLM “gets it” the first time— keep in mind that your prompt is but an address in an infinite ocean of programs, all memorized as a by-product of learning to complete an enormous amount of token sequences.

19.1.3 Deep learning models struggle to learn generalizable programs

The problem with deep learning models isn’t just that they’re limited to blindly reapplying patterns they’ve memorized at training time or that they’re highly sensitive to how their input is presented. Even if you just need to query and apply a well-known program, and you know exactly how to address this program in latent space, you still face a major issue: the programs memorized by deep learning models often don’t generalize well. They will work for some input values and fail for some other input values. This is especially true for programs that encode any kind of discrete logic.

Consider the problem of adding two numbers, represented as character sequences— like “4 3 5 7 + 8 9 3 6.” Try training a Transformer on hundreds of thousands of such digit pairs: you will reach a very high accuracy. Very high, but not 100%—you will keep regularly seeing incorrect answers, because the Transformer doesn’t manage to encode the exact addition algorithm (you know, the one you learned in primary school). It is instead guessing the output by interpolating between the data points it has seen at training time. This applies to state-of-the-art LLMs, too—at least those that weren’t explicitly hardcoded to execute snippets like “4357 + 8936” in Python to provide the right answer. They’ve seen enough examples of digit addition that they can add numbers, but they only have about  …

## Related pages elsewhere in the book (hybrid retrieval; use for the `connections` section)
- PDF p.526 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.589 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.612 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.590 — 19 The future of AI › 19.1 The limitations of deep learning
- PDF p.525 — 16 Text generation › 16.4 Going further with LLMs
- PDF p.528 — 16 Text generation › 16.5 Where are LLMs heading next?
- PDF p.615 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols
- PDF p.616 — 19 The future of AI › 19.4 The missing ingredients: Search and symbols

## Retrieved passages from elsewhere in the book (cite the PDF page when you use them)

### PDF p.525 (printed 501) — 16 Text generation › 16.4 Going further with LLMs › 16.4.3 Retrieval Augmented Generation (RAG)
Sticking extra information in the prompt is not just helpful in handling image data; it can be a general way to extend the capabilities of an LLM. One notable example is when using an LLM for search. If we naively compare an LLM to a search engine, it has a couple of fatal flaws:

An LLM will occasionally make things up. It will output false “facts” that were not present in the training data but could be interpolated from the training data. This information can range from misleading to dangerous.

An LLM’s knowledge of the world has a cutoff date—at best, the date the model was pretrained. Training an LLM is quite expensive, and it is not feasible to train continuously on new data. So at some arbitrary point in time, an LLM’s knowledge of the world will just stop.

No one wants to use a search engine that can only tell you about things that happened six months ago. But if we think of an LLM as more like “conversational software” that can handle any sequence data in a prompt, what if we instead used the model as the interface to information retrieved by more traditional search? That’s the idea behind retrieval-augmented generation or RAG.

### PDF p.629 (printed 605) — 20 Conclusions › 20.2 Limitations of deep learning › 20.2 Limitations of deep learning
Building deep learning models is like playing with LEGO bricks: layers can be plugged together to map essentially anything to anything, given that you have appropriate training data available and that the mapping is achievable via a continuous geometric transformation of reasonable complexity.

Here’s the catch, though—this mapping is often not learnable in a way that will generalize. Deep learning models operate like vast, interpolative databases of patterns. Their pattern-matching strength is also their core weakness:

They fundamentally struggle to adapt to novelty. Because their parameters are fixed after training, they can only retrieve or replicate patterns similar to their training data. Faced with inputs significantly outside this familiar distribution—no matter how simple the underlying task—their performance degrades drastically, as they lack mechanisms for fluid generalization beyond their memorized experience. This explains why even large models fail on novel tasks or simple variations of familiar problems—like ARC-AGI tasks.

They’re very sensitive to phrasing and other distractors. Deep learning models exhibit high sensitivity to superficial variations in input presen

### PDF p.511 (printed 487) — 16 Text generation › 16.3 Using a pretrained LLM › 
Though it’s tempting when working with a model that can “talk” to imagine it interpreting our prompt in some sort of human, conversational way, nothing of the sort is going on here. We have just constructed a prompt for which an actual brownie recipe is a more likely continuation than mimicking someone posting on a forum asking for baking help.

You can go much further in constructing prompts. You might prompt a model with some natural language instructions of the role it is supposed to fill, for example, `"You` `are a large language model that gives short, helpful answers to people's questions."` Or you might feed the model a prompt containing a long list of harmful topics that should not be included in any generated responses. If this all sounds a bit hand-wavy and hard to control, that’s a good assessment. Attempting to visit different parts of a model’s distribution through prompting is often useful, but predicting how a model will respond to a given prompt is very difficult. Another well-documented problem faced by LLMs is hallucinations. A model will always say something—there is always a most-likely next token to a given sequence. Finding locations in our LLM distribution th

### PDF p.589 (printed 565) — 19 The future of AI › 19.1 The limitations of deep learning › 19.1.1 Deep learning models struggle to adapt to novelty
Despite frequent claims that LLMs can perform in-context learning to pick up new skills from just a few examples, there is overwhelming evidence that what they’re actually doing is fetching vector functions they’ve memorized during training and reapplying them to the task at hand. By learning to do next-token prediction across a web-sized text dataset, an LLM has collected millions of potentially useful mini text-processing programs, and it can easily be prompted into reusing them on a new problem. But show it something that has no direct equivalent in its training data, and it’s helpless. Take a look at the puzzle in figure 19.1. Did you figure out the solution? Good. It’s not very hard, is it? But today, no state-of-the-art LLM or vision-language model can do this because this particular problem doesn’t directly map to anything they’ve seen at training time—even after having been trained on the entire internet and then some. An LLM’s ability to solve a given problem has nothing to do with problem complexity, and everything to do with familiarity—they will break their teeth on any sufficiently novel problem, no matter how simple. This failure mode even applies to tiny variations o

### PDF p.528 (printed 504) — 16 Text generation › 16.5 Where are LLMs heading next? › 
In the first attempt, our model was hung up on the superfluous detail that each letter has two pages. In the second attempt, the model gets the problem right. This instruction-tuned Gemma model we are working with has already been tuned on math problems like this; you would not get nearly as good results from the “untuned” Gemma model from the last section.

We could extend this idea to a very simple form of chain-of-thought training: 1 Collect a bunch of basic math and reasoning problems and desired answers. 2 Generate (with some randomness) a number of responses. 3 Find all the responses with a correct answer via string parsing. You can prompt

the model to use a specific text marker for the final answer as we did previously. 4 Run supervised fine-tuning on correct responses, including all the intermediate output. 5 Repeat! The previously described process is a reinforcement learning algorithm. Our answer checking acts as the environment, and the generated outputs are the actions the model uses to learn. As with RLHF, in practice you would use a more complex gradient update step to use information from all responses (even the incorrect ones), but the basic principle is the same. 

### PDF p.590 (printed 566) — 19 The future of AI › 19.1 The limitations of deep learning › 
Figure 19.1 An easy yet novel puzzle

model would just repeat the answer it had memorized without paying any attention to the actual numbers in the query, or what the query really meant. Similarly, LLMs struggle to adapt to variations of the Monty Hall problem (see figure 19.2) and will tend to always output the canonical answer to the puzzle, which they’ve seen many times during training, regardless of whether it makes sense in context.

Figure 19.2 A variation of the Monty Hall problem

To note, these specific prompts were patched later on by special-casing them. Today, there are over 25,000 people who are employed full time to provide training data for LLMs by reviewing failure cases and suggesting better answers. LLM maintenance is a constant game of whack-a-mole where failing prompts are patched one at a time,

### PDF p.526 (printed 502) — 16 Text generation › 16.4 Going further with LLMs › 
RAG works by taking an initial user question and doing some form of a query to pull in additional text context. This query can be to a database, a search engine, or anything that can give further information on the question asked by a user. This extra information is then added straight into the prompt. For example, you might construct a prompt like this:

Use the following pieces of context to answer the question. Question: What are some good ways to improve sleep? Context: {text from a medical journal on improving sleep} Answer:

A common approach for looking up relevant information is to use a vector database. To build a vector database, you can use an LLM, or any model, to embed a series of source documents as vectors. The document text will be stored in the database, with the embedding vector used as a key. During retrieval, an LLM can again be used to embed the user query as a vector. The vector database is responsible for searching for key vectors close to the query vector and for surfacing the corresponding text. This might sound a lot like the attention mechanism itself—recall that the terms “query,” “key,” and “value” actually came from database systems.

Surfacing informa

### PDF p.527 (printed 503) — 16 Text generation › 16.4 Going further with LLMs › 
you didn’t even need examples; you could simply prompt the model with the phrase “Let’s think step by step” and get better output.

Since these discoveries, there has been heavy interest in directly training LLMs to get better at chain-of-thought reasoning. Models like OpenAI’s o1 and DeepSeek’s r1 have made headlines by showing significant strides in math and coding problems by training a model to “think out loud” on difficult questions. The approach for this chain-of-thought fine-tuning is very similar to RLHF. We will first train the model on a few supervised examples of “showing your work” on a math problem and arriving at a correct answer. Next, we will prompt the model with a new math question and check whether the model got the final answer correct. Finally, we use these new generated outputs to further tune the model’s weights. Let’s try this out with the Gemma model. We can write out our own word problem and turn on random sampling so we get a somewhat random response each time:
