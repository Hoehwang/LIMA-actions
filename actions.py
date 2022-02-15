# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

import pandas as pd
import random
import numpy as np
import re
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

res = pd.read_csv("./actions/RESPONSE_EXP_LIMA.csv", encoding = 'utf8')
syn = pd.read_csv("./actions/SYN.csv", encoding = 'utf8')
ac_ta = pd.read_csv('./actions/LIMA-ACTION-TABLE.csv', encoding = 'utf8') # 질병별 증상 정보에 대한 mapping table
ac_di = pd.read_csv('./actions/LIMA-ACTION-DISEASE.csv', encoding = 'utf8') # 질병별 세부 정보(원인 증상 등)

#ac_ta 전처리(Nan 제거 및 int형으로)
ac_ta = ac_ta.fillna(0)
ac_ta_2 = ac_ta.iloc[:,3:].astype(int)
ac_ta_1 = ac_ta.iloc[:,:3]
ac_ta = pd.concat([ac_ta_1, ac_ta_2], axis=1)

default_body = '피부'

class ActionRephraseResponse(Action):
        
    # 액션에 대한 이름을 설정
    def name(self) -> Text:
        return "action_rephrase_medical"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print(tracker.latest_message['entities'])
        print(tracker.get_intent_of_latest_message())

        self.symptom_dic = {}
        self.body_dic = {}
        self.body = ''
        self.disease_dic = {}
        self.disease = ''
        self.type_mapper = {} #entity의 type을 알려줌

        self.entity = [a.get("entity") for a in tracker.latest_message["entities"]]
        self.intent = tracker.get_intent_of_latest_message()
        self.dis, self.eve = self.intent.split('_')

        if self.eve != 'ALL-FEATURE' and self.eve != 'DISEASE': #intent가 why_lump와 같이 1개 짜리 증상표현일 경우, 해당 eve(lump)를 entity에 추가한다.
            self.entity.append(self.eve)

        for ent in self.entity:
            if syn[syn["entity"]==ent]["type"].item() == "SYMPTOM":
                self.symptom_dic[ent] = syn[syn["entity"]==ent]["norm"].item()
                self.type_mapper[ent] = "SYMPTOM"
            elif syn[syn["entity"]==ent]["type"].item() == "BODY":
                self.body_dic[ent] = syn[syn["entity"]==ent]["norm"].item()
                self.type_mapper[ent] = "BODY"
            else: #DISEASE
                self.disease_dic[ent] = syn[syn["entity"]==ent]["norm"].item()
                self.type_mapper[ent] = "DISEASE"
        
        #body와 disease의 경우, 1개만을 요할 경우에 빠른 처리를 위해 변수에 따로 저장
        if self.body_dic.keys():
            self.body = list(self.body_dic.keys())[0]
        if self.disease_dic.keys():
            self.disease = list(self.disease_dic.keys())[0]


        if self.eve == "DISEASE" and self.disease:
            nlg_form = random.choice(res[res["intent"]==self.intent]["response"].values[0].split(' / '))
            nlg_form = nlg_form.replace('<DISEASE_FEATURE>',self.disease_dic[self.disease])
            dispatcher.utter_message(text=nlg_form)
            dispatcher.utter_message(text=res[res["intent"]==self.intent]["send_expression"].item())
            expression = ac_di[ac_di["entity"]==self.disease][self.dis].item()

            dispatcher.utter_message(text=expression.split('#')[0])

            dispatcher.utter_message(text=res[res["intent"]==self.intent]["send_link"].item())
            dispatcher.utter_message(link = ac_di[ac_di["entity"]==self.disease]["link"].item())
            dispatcher.utter_message(text=res[res["intent"]==self.intent]["utter_ask_more"].item())
        
        elif self.eve == "ALL-FEATURE":
            nlg_form = random.choice(res[res["intent"]==self.dis]["response"].values[0].split(' / '))
            slot_form = self.slot_maker()
            nlg_form = nlg_form.replace('<SLOT>',slot_form)
            dispatcher.utter_message(text=nlg_form)
            

            #증상에 따른 병명 추출 및 병명별 dis에 따른 특징 기술
            res_disease_ls = self.disease_finder()
            norm_res_disease_ls = list(map(lambda x: syn[syn["entity"]==x]["norm"].item(), res_disease_ls))
            if len(norm_res_disease_ls) == 0:
                dispatcher.utter_message(text=res[res["intent"]==self.dis]["entityless"].item())
                return    
            dispatcher.utter_message(text=res[res["intent"]==self.dis]["send_link"].item())
            dispatcher.utter_message(text="관련 병명: " + '/'.join(norm_res_disease_ls))
            
            #추출된 질병이 3개 이상인지 미만인지에 따라
            if len(res_disease_ls) > 2: #3개 이상
                for i, d in enumerate(res_disease_ls[:3]):
                    dispatcher.utter_message(text='{}) '.format(i+1)+norm_res_disease_ls[i])
                    expression = ac_di[ac_di["entity"]==d][self.dis].item()
                    dispatcher.utter_message(text=expression.split('#')[0]+'\n')
                    link = ac_di[ac_di["entity"]==d]["link"].item()
                    dispatcher.utter_message(text='관련 링크: '+link+'\n')
                    
            else: #3개 미만
                for i, d in enumerate(res_disease_ls):
                    dispatcher.utter_message(text='{}) '.format(i+1)+norm_res_disease_ls[i])
                    expression = ac_di[ac_di["entity"]==d][self.dis].item()
                    dispatcher.utter_message(text=expression.split('#')[0]+'\n')
                    link = ac_di[ac_di["entity"]==d]["link"].item()
                    dispatcher.utter_message(text=link)
            dispatcher.utter_message(text=res[res["intent"]==self.dis]["utter_ask_more"].item())

        else: #all-feature가 아닌 인텐트(증상 엔티티 1개 짜리-그러나 아닐 수도..)
            if len(list(self.symptom_dic.keys())) > 1:
                nlg_form = random.choice(res[res["intent"]==self.dis]["response"].values[0].split(' / '))
                slot_form = self.slot_maker()
                nlg_form = nlg_form.replace('<SLOT>',slot_form)
                dispatcher.utter_message(text=nlg_form)

                #증상에 따른 병명 추출 및 병명별 dis에 따른 특징 기술
                res_disease_ls = self.disease_finder()
                norm_res_disease_ls = list(map(lambda x: syn[syn["entity"]==x]["norm"].item(), res_disease_ls))
                if len(norm_res_disease_ls) == 0:
                    dispatcher.utter_message(text=res[res["intent"]==self.dis]["entityless"].item())
                    return    
                dispatcher.utter_message(text=res[res["intent"]==self.dis]["send_link"].item())
                dispatcher.utter_message(text="관련 병명: " + '/'.join(norm_res_disease_ls))
                
                #추출된 질병이 3개 이상인지 미만인지에 따라
                for i, d in enumerate(res_disease_ls):
                    if i > 2: break
                    dispatcher.utter_message(text='{}) '.format(i+1)+norm_res_disease_ls[i])
                    expression = ac_di[ac_di["entity"]==d][self.dis].item()
                    dispatcher.utter_message(text=expression.split('#')[0]+'\n')
                    link = ac_di[ac_di["entity"]==d]["link"].item()
                    dispatcher.utter_message(text='관련 링크: '+link+'\n')
                dispatcher.utter_message(text=res[res["intent"]==self.dis]["utter_ask_more"].item())
            
            else: #증상 엔티티 1개 짜리
                nlg_form = random.choice(syn[syn["entity"]==self.eve][self.dis].values[0].split(' / '))
                if self.body:
                    nlg_form = nlg_form.replace('<BODY_FEATURE>',self.body_dic[self.body]+self.get_josa(self.body_dic[self.body][-1], self.eve))
                else: 
                    nlg_form = nlg_form.replace('<BODY_FEATURE> ', '')
                dispatcher.utter_message(text=nlg_form)

                #증상에 따른 병명 추출 및 병명별 dis에 따른 특징 기술
                res_disease_ls = self.disease_finder()
                print(res_disease_ls)
                norm_res_disease_ls = list(map(lambda x: syn[syn["entity"]==x]["norm"].item(), res_disease_ls))
                if len(norm_res_disease_ls) == 0:
                    dispatcher.utter_message(text=res[res["intent"]==self.dis]["entityless"].item())
                    return    
                dispatcher.utter_message(text=res[res["intent"]==self.dis]["send_link"].item())
                dispatcher.utter_message(text="관련 병명: " + '/'.join(norm_res_disease_ls))
                
                #추출된 질병이 3개 이상인지 미만인지에 따라
                for i, d in enumerate(res_disease_ls):
                    if i > 2: break
                    dispatcher.utter_message(text='{}) '.format(i+1)+norm_res_disease_ls[i])
                    expression = ac_di[ac_di["entity"]==d][self.dis].item()
                    dispatcher.utter_message(text=expression.split('#')[0]+'\n')
                    link = ac_di[ac_di["entity"]==d]["link"].item()
                    dispatcher.utter_message(text='관련 링크: '+link+'\n')
                dispatcher.utter_message(text=res[res["intent"]==self.dis]["utter_ask_more"].item())

    def get_josa(self, syl, ent): #ent는 다음에 올 술어 표현에 해당하는 증상 entity 종류
        josa_mapper = {'<가>':['이', '가']}
        if syn[syn["entity"]==ent]["body-josa"].item()[0] != '<':
            return syn[syn["entity"]==ent]["body-josa"].item()
        else:
            if self.get_jong(syl) == '': #무종성
                return josa_mapper[syn[syn["entity"]==ent]["body-josa"].item()][1]
            else: #유종성
                return josa_mapper[syn[syn["entity"]==ent]["body-josa"].item()][0]

    def get_jong(self, syllable):
        codas = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

        syllable_number = ord(syllable) - ord('가')
        return codas[syllable_number % 28]

    def slot_maker(self): #NLG를 수행할 SLOT을 만드는 함수
        slot_form = ''
        # ent_len = len(self.entity) #엔티티 총 개수
        symp_len = len(self.symptom_dic.keys())
        # body_len = len(self.symptom_dic.keys())
        symp_cnt = 1 #symptom 표현의 개수를 세기 위한 counter

        body_ent = 0 #body가 앞에 나왔는지
        last_body_syl = ''
        for ent in self.entity:
            if self.type_mapper[ent] == 'BODY':
                slot_form += self.body_dic[ent]
                body_ent = 1
                last_body_syl = self.body_dic[ent][-1]
            else:
                if body_ent == 1:
                    slot_form += self.get_josa(last_body_syl, ent) + ' '
                    body_ent = 0
                if symp_cnt != symp_len:
                    slot_form += self.symptom_dic[ent] + '고 '
                    symp_cnt += 1
                else:
                    slot_form += self.symptom_dic[ent] + '는'
                    symp_cnt += 1
            

        return slot_form
    
    def disease_finder(self): # 증상들로부터 병명을 추출하는 함수
        query_form = ''
        temp_query = []
        for ent in self.entity:
            temp_query.append(ent.replace('-','_') + '==1')
        
        if len(temp_query)>1:
            query_form = ' & '.join(temp_query)
        else:
            query_form = temp_query[0]

        temp_ac_ta = ac_ta.query(query_form)
        res_disease_ls = temp_ac_ta["영문명"].to_list()

        return res_disease_ls

#들어온 인텐트, 엔티티에 대해 다음과 같이 처리
#1) 인텐트 인식
    #1-1) 인텐트 파싱: '_'기준으로 둘로 나눔(DIS/EVE), DISEASE, SYMPTOM(DISEASE아니면 SYMPTOM)
    #1-2) DISEASE는 res에서 답변 처리
        #1-2-1) ac_di에서 해당 disease 찾은 후 관련 내용(담화소 정보)에 대해 처음 몇 문장 출력(혹은 띄어쓰기 2개로 split 후 첫 덩이 출력)
    #1-3) SYMPTOM은 엔티티 추출(ENT_LIST), INTENT의 EVE가 ALL-FEATURE인지 파악, 아니면 ENT_LIST에 추가, SYN에서 TYPE 파악가능?(BODY인지 아닌지)
        #1-3-1) ALL-FEATURE: SLOT을 RES에서 가져옴, ENT를 NORM값으로 치환하며, 그 다음 <SLOT>에 BODY에-ENT고-BODY에-ENT고-ENT는 순으로 뿌려줌..? 아니면 등장한 순서로..?
        #1-3-2) 일반 INTENT:
            #1-3-2-1) BODY아닌 ENT 2개 이상: all-feature와 동일하게 처리
            #1-3-2-2) BODY아닌 ENT 1개: syn에서 슬롯 가져옴. default_body + ent