#Catherine Magee
#Question: Write a function that takes 2-3 paragraphs input-text and list of questions as input.
# It will return the answers to the questions from the input-text as the output.
# It should get the answers from the input-text only. If the question is outside the scope of the input-text,
# it should give a generic answer "out of scope".

#If you would like to see how Bert answered my questions and my paragraph uncomment the variables questions and context!

# questions = [
#     "What organization is the IPCC a part of?",
#     "What UN organizations established the IPCC?",
#     "What does the UN want to stabilize?",
#     "What does climate change mean?"
# ]
# context = (
#     "The Intergovernmental panel on climate change (IPCC) is a scientific intergovernmental body under"
#     " the auspices of the United Nations, set up at the request of member governments. It was first established in 1988"
#     "by two United Nations organizations, the World Meteorological Organization (WMO) and the United Nations Enviroment "
#     "programme (UNEP), and later endorsed by the United Nations General Assembly through Resolution 43/53. Membership "
#     "of the IPCC is open to all members of the WMO and UNEP. The IPCC produces reports that support the United Nations"
#     "Framework Convention on Climate Change (UNFCCC), which is the main international treaty on climate change. The "
#     "ultimate objective of the UNFCCC is to stabilize greenhouse gas concentrations in the atmosphere at a level that"
#     "would prevent dangerous anthropogenic interference with the climate system. IPCC reports cover the scientific,"
#     "technical and socio-economic information relevant to understanding the scientific basis of risk of human-induced"
#     "climate change, its potential impacts and options for adaptation and mitigation."
# )

questions = []
questions = [item for item in input("Please input your questions. Do not worry about formatting just don't forget the ? symbol ").split("?")]

context = input("Please past your paragraph here. Do not worry about formatting!  ")

from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')


#convert strings to bert language using tokenizers
# 101 = 'CLS' start of a sequence
# 102 = 'SEP' end of a sequence/seperator
# what goes into bert -> [CLS] <context> [SEP] <question> [SEP] [PAD] [PAD]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

for i in range(0, len(questions)):
    tokenizer.encode(questions[i], truncation = True, padding = True)

from transformers import pipeline
nlp = pipeline('question-answering',model = model, tokenizer = tokenizer)

for j in range(0, len(questions) - 1):
    nlp_test = nlp({
        'question': questions[j],
        'context': context
    })
    # the order [score, context[start], context[end], answer]
    results = nlp_test.values()
    list = []
    for y in results:
        list.append(y)
    if list[0] <= 0.009:
        print("Your question is out of Bert's scope!")
    else:
        print("The answer to question " + str(j + 1) + " is " + list[3])








