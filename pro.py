from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import defaultdict
from networkx import Graph, pagerank

class SummarizerApp(App):
    def summarize_text(self, text, ratio=0.2):
        """
        This function summarizes a given text using extractive summarization with
        named entity recognition (NER), sentence weighting based on position, named entities,
        and sentence centrality.

        Args:
            text: The text to be summarized (large string).
            ratio: The ratio of the original text length to use in the summary (default 0.2).

        Returns:
            A string containing the summarized text.
        """
        try:
            nltk.download('stopwords')
        except LookupError:
            pass
        try:
            nlp = spacy.load("en_core_web_sm")  # Load spaCy model (English)
        except ImportError:
            print("spaCy NER not available (requires installing 'en_core_web_sm' model).")

        sentences = nltk.sent_tokenize(text)  # Use nltk for sentence tokenization

        # Calculate number of sentences for summary based on ratio
        num_sentences = int(len(sentences) * ratio)

        # Implement scoring system (TF-IDF + Positional Weighting + Named Entity Weighting + Sentence Centrality)
        stop_words = nltk.corpus.stopwords.words('english')  # Use stop words removal
        vectorizer = TfidfVectorizer(stop_words=stop_words)  # Include stop words removal
        tf_idf_matrix = vectorizer.fit_transform([sentences[i] for i in range(len(sentences))])
        sentence_scores = tf_idf_matrix.sum(axis=1).A.flatten()

        # Positional weighting (increase score for beginning/end of paragraphs)
        for i, sentence in enumerate(sentences):
            if i == 0 or i == len(sentences) - 1:  # First or last sentence of paragraph
                sentence_scores[i] *= 1.2  # Increase score by 20%

        # Named Entity Recognition (NER)
        if nlp is not None:
            for i, sentence in enumerate(sentences):
                doc = nlp(sentence)
                for entity in doc.ents:
                    if entity.label_ in ("PERSON", "GPE", "LOC"):  # People, organizations, locations
                        sentence_scores[i] *= 1.1  # Slightly increase score for sentences with named entities

        # Sentence Centrality (using TextRank)
        def textrank(sentences):
            graph = Graph()
            for i in range(len(sentences) - 1):
                for j in range(i + 1, len(sentences)):
                    # More sophisticated overlap check (considering synonyms or word stems) can be implemented here
                    if any(word in sentences[j] for word in sentences[i].split()):  # Overlap check
                        graph.add_edge(i, j)
            return pagerank(graph)  # Use NetworkX pagerank function

        try:
            sentence_centralities = textrank(sentences)
            for i, score in enumerate(sentence_centralities):
                sentence_scores[i] *= score
        except ImportError:
            print("NetworkX not available (required for TextRank).")

        # Sort sentences by score and select the top ones for summary
        sorted_sentences = sorted(enumerate(sentence_scores), key=lambda x: x[1], reverse=True)[:num_sentences]

        # Generate summary by joining the selected sentences
        summary = ". ".join([sentences[i] for i, score in sorted_sentences])

        return summary

    def build(self):
        
        layout = BoxLayout(orientation='vertical')

        # Input text
        input_label = Label(text="Enter the text you want to summarize:")
        self.input_text = TextInput(size_hint_y=None, height=200, multiline=True)
        layout.add_widget(input_label)
        layout.add_widget(self.input_text)

        # Output text
        output_label = Label(text="Summary:")
        self.output_text = TextInput(size_hint_y=None, height=200, multiline=True, readonly=True)
        output_scroll = ScrollView(size_hint=(1, None), size=(400, 200))
        output_scroll.add_widget(self.output_text)
        layout.add_widget(output_label)
        layout.add_widget(output_scroll)

        # Button to trigger summarization
        button = Button(text="Summarize", size_hint_y=None, height=50)
        button.bind(on_press=self.summarize)
        layout.add_widget(button)

        return layout

    def summarize(self, instance):
        input_text = self.input_text.text
        summary = self.summarize_text(input_text)
        self.output_text.text = summary

if __name__ == '__main__':
    SummarizerApp().run()


