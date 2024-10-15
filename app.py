from flask import Flask, request, jsonify, render_template
from langdetect import detect_langs
from transformers import pipeline
from collections import defaultdict
from translate import Translator

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sentiment-analysis')
def index():
    return app.send_static_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    post = data['post']

    # Load the pre-trained sentiment analysis model from transformers
    sentiment_analysis = pipeline("sentiment-analysis")

    # Analyze sentiment using BERT
    result = sentiment_analysis(post)[0]
    sentiment = result['label']
    return jsonify({'sentiment': sentiment})


@app.route('/language-detection')
def index1():
    return render_template('page_a.html')


@app.route('/pageB', methods=['GET', 'POST'])
def page_b():
    data = None
    detected_languages = None
    if request.method == 'GET':
        data = request.args.get('data')
        if data:
            # Perform language detection
            langs = detect_langs(data)
            detected_languages = defaultdict(float)
            for lang in langs:
                detected_languages[lang.lang] += lang.prob
            # Normalize probabilities and sort by probability
            total_prob = sum(detected_languages.values())
            detected_languages = {lang: prob/total_prob for lang, prob in detected_languages.items()}
            detected_languages = {lang_name(lang): prob for lang, prob in detected_languages.items()}
    return render_template('page_b.html', data=data, detected_languages=detected_languages)


def lang_name(lang_code):
    # Function to map language code to language name
    lang_names = {
        'af': 'Afrikaans',
        'sq': 'Albanian',
        'am': 'Amharic',
        'ar': 'Arabic',
        'hy': 'Armenian',
        'az': 'Azerbaijani',
        'eu': 'Basque',
        'be': 'Belarusian',
        'bn': 'Bengali',
        'bs': 'Bosnian',
        'bg': 'Bulgarian',
        'ca': 'Catalan',
        'ceb': 'Cebuano',
        'ny': 'Chichewa',
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
        'co': 'Corsican',
        'hr': 'Croatian',
        'cs': 'Czech',
        'da': 'Danish',
        'nl': 'Dutch',
        'en': 'English',
        'eo': 'Esperanto',
        'et': 'Estonian',
        'tl': 'Filipino',
        'fi': 'Finnish',
        'fr': 'French',
        'fy': 'Frisian',
        'gl': 'Galician',
        'ka': 'Georgian',
        'de': 'German',
        'el': 'Greek',
        'gu': 'Gujarati',
        'ht': 'Haitian Creole',
        'ha': 'Hausa',
        'haw': 'Hawaiian',
        'iw': 'Hebrew',
        'he': 'Hebrew',  # Alias for Hebrew
        'hi': 'Hindi',
        'hmn': 'Hmong',
        'hu': 'Hungarian',
        'is': 'Icelandic',
        'ig': 'Igbo',
        'id': 'Indonesian',
        'ga': 'Irish',
        'it': 'Italian',
        'ja': 'Japanese',
        'jv': 'Javanese',
        'kn': 'Kannada',
        'kk': 'Kazakh',
        'km': 'Khmer',
        'rw': 'Kinyarwanda',
        'ko': 'Korean',
        'ku': 'Kurdish (Kurmanji)',
        'ky': 'Kyrgyz',
        'lo': 'Lao',
        'la': 'Latin',
        'lv': 'Latvian',
        'lt': 'Lithuanian',
        'lb': 'Luxembourgish',
        'mk': 'Macedonian',
        'mg': 'Malagasy',
        'ms': 'Malay',
        'ml': 'Malayalam',
        'mt': 'Maltese',
        'mi': 'Maori',
        'mr': 'Marathi',
        'mn': 'Mongolian',
        'my': 'Myanmar (Burmese)',
        'ne': 'Nepali',
        'no': 'Norwegian',
        'or': 'Odia (Oriya)',
        'ps': 'Pashto',
        'fa': 'Persian',
        'pl': 'Polish',
        'pt': 'Portuguese',
        'pa': 'Punjabi',
        'ro': 'Romanian',
        'ru': 'Russian',
        'sm': 'Samoan',
        'gd': 'Scots Gaelic',
        'sr': 'Serbian',
        'st': 'Sesotho',
        'sn': 'Shona',
        'sd': 'Sindhi',
        'si': 'Sinhala',
        'sk': 'Slovak',
        'sl': 'Slovenian',
        'so': 'Somali',
        'es': 'Spanish',
        'su': 'Sundanese',
        'sw': 'Swahili',
        'sv': 'Swedish',
        'tg': 'Tajik',
        'ta': 'Tamil',
        'tt': 'Tatar',
        'te': 'Telugu',
        'th': 'Thai',
        'tr': 'Turkish',
        'tk': 'Turkmen',
        'uk': 'Ukrainian',
        'ur': 'Urdu',
        'ug': 'Uyghur',
        'uz': 'Uzbek',
        'vi': 'Vietnamese',
        'cy': 'Welsh',
        'xh': 'Xhosa',
        'yi': 'Yiddish',
        'yo': 'Yoruba',
        'zu': 'Zulu'
    }
    return lang_names.get(lang_code, lang_code)



@app.route('/translate', methods=['GET', 'POST'])
def translate():
    if request.method == 'GET':
        return render_template('index1.html')
    elif request.method == 'POST':
        data = request.json
        if data:
            text = data.get('text')
            target_language = data.get('target_language')
            if text and target_language:
                try:
                    translator = Translator(to_lang=target_language)
                    translated_text = translator.translate(text)
                    return jsonify({'translation': translated_text}), 200
                except Exception as e:
                    return jsonify({'error': str(e)}), 500
        return jsonify({'error': 'Invalid request data'}), 400

if __name__ == '__main__':
    app.run(debug=True)
