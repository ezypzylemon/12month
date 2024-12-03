# 필요한 라이브러리 임포트
import os
import re
import requests
import pandas as pd
from wordcloud import WordCloud  # 워드클라우드 생성을 위한 모듈
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지를 생성하기 위해 Agg 백엔드 사용
import matplotlib.pyplot as plt  # 시각화를 위한 모듈
from collections import Counter  # Counter 클래스 임포트
from konlpy.tag import Okt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class KeywordGenerator:
    def __init__(self):
        # .env 파일 로드
        load_dotenv()
        
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"API 키 확인: {api_key[:5]}...") # API 키의 처음 5자리만 출력
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 환경 변수에 설정되어 있지 않습니다.")
            
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=1.1,
            openai_api_key=api_key
        )
        
        # 프롬프트 템플릿 생성
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", """다음 키워드와 의미적으로 비슷한 단어 20개를 생성해주세요.
            각 단어는 숫자와 점으로 시작하고 한 줄에 하나씩 표시해주세요.
            키워드: {keyword}""")
        ])
        
        # LLMChain 생성
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        # 결과 저장할 파일명 설정
        self.output_file = "keyword_results.csv"
        
        # 초기 키워드 입력 받기
        self.keyword = input("비슷한 키워드를 생성할 단어를 입력하세요: ").strip()
        if not self.keyword:
            raise ValueError("입력된 키워드가 없습니다.")

        # 네이버 API 인증 정보 설정
        self.client_id = 'UGWFRw8nJK3z1B_irpX9'
        self.client_secret = 'tGq_GW6jBz'

        # 결과 저장을 위한 폴더 생성
        self.folder_name = "naver_kin_results"
        os.makedirs(self.folder_name, exist_ok=True)

        # 워드클라우드 이미지 저장을 위한 폴더 생성
        self.wordcloud_folder = "wordcloud_images"
        if not os.path.exists(self.wordcloud_folder):
            os.makedirs(self.wordcloud_folder)

        # 키워드별 명사 빈도수를 저장할 딕셔너리
        self.keyword_noun_counts = {}
        
        # 키워드 DataFrame 초기화
        self.keywords_df = None

    def generate_similar_keywords(self, keyword):
        """
        LangChain을 사용하여 비슷한 키워드 생성
        """
        try:
            # LLMChain을 통해 응답 생성
            response = self.chain.run(keyword=keyword)
            # 숫자. 형식의 라인만 추출
            similar_words = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if line and line[0].isdigit() and ". " in line:
                    word = line.split(". ", 1)[1].strip()
                    similar_words.append(word)
            return similar_words[:20]  # 최대 20개만 반환
        except Exception as e:
            print(f"키워드 생성 중 오류 발생: {e}")
            return []

    def search_naver_kin(self, keywords_df):
        """네이버 지식iN 검색 수행"""
        url = 'https://openapi.naver.com/v1/search/kin.json'
        headers = {
            'X-Naver-Client-Id': self.client_id,
            'X-Naver-Client-Secret': self.client_secret
        }

        for idx, row in keywords_df.iterrows():
            query = row.iloc[0]

            params = {
                "query": query,
                "display": 100
            }

            response = requests.get(url, headers=headers, params=params)
            result = response.json()

            temp_df = pd.DataFrame(result['items'])
            temp_df['keyword'] = query

            file_path = os.path.join(self.folder_name, f"kin_{idx+1}. {query}.csv")
            temp_df.to_csv(file_path, index=False, encoding='utf-8-sig')

            print(f"키워드 '{query}'의 결과 저장 완료: {file_path}")

    def preprocess_data(self):
        """데이터 전처리 수행"""
        for filename in os.listdir(self.folder_name):
            if filename.startswith('kin_') and filename.endswith('.csv'):
                file_path = os.path.join(self.folder_name, filename)
                df = pd.read_csv(file_path, encoding='utf-8-sig')
                
                if 'keyword' in df.columns and not df['keyword'].empty:
                    query = df['keyword'].values[0]
                    query = re.sub(r'^\d+\.\s*', '', query)
                else:
                    # 파일명에서 인덱스와 키워드 추출
                    match = re.match(r'kin_(\d+)\.\s+(.+)\.csv', filename)
                    if match:
                        query = match.group(2)
                    else:
                        continue
                
                if 'title' in df.columns:
                    df['title'] = df['title'].fillna('')  # NaN 값을 빈 문자열로 대체
                    df['title'] = df['title'].str.replace(query, '', regex=False)
                    df['title'] = df['title'].str.strip()
                    df['title'] = df['title'].str.replace(r'\s+', ' ', regex=True)
                    df['title'] = df['title'].str.replace(r'<b>|</b>', '', regex=True)
                
                if 'description' in df.columns:
                    df['description'] = df['description'].fillna('')
                    df['description'] = df['description'].str.strip()
                    df['description'] = df['description'].str.replace(r'\s+', ' ', regex=True)
                    df['description'] = df['description'].str.replace(r'<b>|</b>', '', regex=True)
                
                # 파일명에서 인덱스와 키워드 추출
                match = re.match(r'kin_(\d+)\.\s+(.+)\.csv', filename)
                if match:
                    idx = match.group(1)
                    keyword = match.group(2)
                    output_filename = f"preprocessed_kin_{idx}. {keyword}.csv"
                    output_path = os.path.join(self.folder_name, output_filename)
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    print(f"{filename}의 전처리가 완료되었습니다: {output_filename}")

    def extract_nouns(self):
        """명사 추출 및 태깅"""
        okt = Okt()
        processed_files = set()
        files_without_title = []

        for filename in os.listdir(self.folder_name):
            if not filename.startswith('preprocessed_kin_') or not filename.endswith('.csv'):
                continue
                
            if filename in processed_files or filename.startswith('morphs_'):
                continue
                
            file_path = os.path.join(self.folder_name, filename)
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            
            # 파일명에서 인덱스 추출 (preprocessed_kin_숫자. 키워드.csv 형식)
            match = re.search(r'kin_(\d+)\.', filename)
            if not match:
                print(f"파일명 {filename}에서 인덱스를 추출할 수 없습니다.")
                continue
                
            file_idx = int(match.group(1)) - 1
            if file_idx >= len(self.keywords_df):
                print(f"파일 인덱스 {file_idx}가 키워드 데이터프레임의 범위를 벗어났습니다.")
                continue
                
            current_keyword = self.keywords_df.iloc[file_idx, 0]
            
            if 'title' in df.columns:
                title_nouns_list = []
                nouns = []
                for title in df['title']:
                    if pd.isna(title):  # NaN 값 처리
                        title_nouns = []
                    else:
                        title_nouns = okt.nouns(str(title))
                    title_nouns_list.append(title_nouns)
                    nouns.extend(title_nouns)
                
                df['title_nouns'] = title_nouns_list
                nouns_count = Counter(nouns)
                if nouns_count:  # 빈도수가 있는 경우에만 저장
                    self.keyword_noun_counts[current_keyword] = nouns_count.most_common(10)
                
                # 파일명에서 인덱스와 키워드 추출하여 새로운 형식으로 저장
                match = re.match(r'preprocessed_kin_(\d+)\.\s+(.+)\.csv', filename)
                if match:
                    idx = match.group(1)
                    keyword = match.group(2)
                    output_filename = f"morphs_kin_{idx}. {keyword}.csv"
                    output_path = os.path.join(self.folder_name, output_filename)
                    df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                processed_files.add(filename)
                print(f"{filename}의 명사 추출이 완료되어 {output_filename}으로 저장되었습니다.")
                print(f"상위 10개 명사: {nouns_count.most_common(10)}")
            else:
                files_without_title.append(filename)
                print(f"{filename} 파일에 'title' 열이 없습니다.")

    def generate_wordcloud(self):
        """워드클라우드 생성"""
        for filename in os.listdir(self.folder_name):
            if filename.startswith('morphs_kin_'):
                try:
                    file_path = os.path.join(self.folder_name, filename)
                    df = pd.read_csv(file_path, encoding='utf-8-sig')
                    
                    if 'keyword' not in df.columns or df.empty:
                        print(f"{filename} 파일에 키워드가 없거나 파일이 비어있습니다.")
                        continue
                        
                    current_keyword = df['keyword'].iloc[0]
                    
                    if current_keyword in self.keyword_noun_counts and self.keyword_noun_counts[current_keyword]:
                        # 빈도수 딕셔너리 생성
                        freq_dict = dict(self.keyword_noun_counts[current_keyword])
                        
                        if not freq_dict:  # 빈도수 딕셔너리가 비어있는 경우
                            print(f"{current_keyword}에 대한 명사 빈도수가 없어서 워드클라우드를 생성할 수 없습니다.")
                            continue
                            
                        wordcloud = WordCloud(
                            font_path='/System/Library/Fonts/AppleSDGothicNeo.ttc',
                            background_color='white', 
                            width=800,
                            height=600
                        ).generate_from_frequencies(freq_dict)

                        # 파일명에서 특수문자 제거
                        safe_keyword = re.sub(r'[^\w\s-]', '', current_keyword)
                        wordcloud_filename = f"wordcloud_{safe_keyword}.png"
                        wordcloud_path = os.path.join(self.wordcloud_folder, wordcloud_filename)
                        
                        plt.figure(figsize=(10, 6))
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title(f"Word Cloud for {current_keyword}")
                        
                        plt.savefig(wordcloud_path, bbox_inches='tight', pad_inches=0, dpi=300)
                        plt.close()
                        
                        print(f"워드클라우드가 {wordcloud_path}로 저장되었습니다.")
                    else:
                        print(f"{current_keyword}에 대한 명사 빈도수가 0이어서 워드클라우드를 생성할 수 없습니다.")
                except Exception as e:
                    print(f"워드클라우드 생성 중 오류 발생: {e}")

    def run(self):
        """전체 프로세스 실행"""
        print("키워드 생성을 시작합니다.")
        
        similar_keywords = self.generate_similar_keywords(self.keyword)
        if similar_keywords:
            print(f"\n'{self.keyword}'와 의미적으로 비슷한 키워드:\n")
            
            self.keywords_df = pd.DataFrame({
                '키워드': similar_keywords
            })
            
            for i, word in enumerate(similar_keywords, 1):
                print(f"{i}. {word}")
            
            self.keywords_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
            print(f"\n결과가 {self.output_file}에 저장되었습니다.")

            # 네이버 지식iN 검색
            self.search_naver_kin(self.keywords_df)
            print("모든 키워드에 대한 결과 저장 완료")

            # 데이터 전처리
            self.preprocess_data()
            print("모든 검색 결과 파일의 전처리가 완료되었습니다.")

            # 명사 추출
            self.extract_nouns()

            # 워드클라우드 생성
            self.generate_wordcloud()
            print("모든 morphs_preprocessed_kin 파일의 워드클라우드 생성이 완료되었습니다.")
        else:
            print("키워드 생성을 실패했습니다.")

if __name__ == "__main__":
    generator = KeywordGenerator()
    generator.run()