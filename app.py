import streamlit as st
import spacy
import nltk
from nltk.corpus import wordnet
from graphviz import Digraph
import os

# ==========================================
# 0. 环境配置 (防止本地 Graphviz 报错)
# ==========================================
# os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==========================================
# 1. 资源加载与配置
# ==========================================
st.set_page_config(page_title="NLP to UML (Thesis Edition)", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    div[data-testid="stGraphvizChart"] {
        text-align: center;
        border: 1px solid #e6e6e6;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'generated_classes' not in st.session_state: st.session_state['generated_classes'] = {}
if 'graph_dot' not in st.session_state: st.session_state['graph_dot'] = None

@st.cache_resource
def load_nlp_resources():
    # 1. 下载 WordNet (用于 Ontology Check)
    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    
    # 2. 加载 Spacy 模型 (针对 Streamlit Cloud 的终极防御写法)
    try:
        # 尝试直接 import (如果 requirements.txt URL 生效)
        import en_core_web_sm
        return en_core_web_sm.load()
    except ImportError:
        # 如果 import 失败，尝试标准加载
        try:
            return spacy.load("en_core_web_sm")
        except:
            return None

nlp = load_nlp_resources()

# ==========================================
# 2. 核心算法逻辑 (Phase 3: Extraction Rules)
# ==========================================
class ThesisUMLSystem:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        self.classes = {}
        self.relationships = []
        # 过滤掉非实质性动词
        self.ignored_verbs = {"be", "have", "include", "consist", "contain", "involve"}

    def check_ontology(self, word):
        """
        利用 WordNet 验证提取的词是否具备名词实体的语义，
        避免提取出 "system", "process" 等抽象泛词（论文创新点之一）。
        """
        try:
            synsets = wordnet.synsets(word)
            if not synsets: return True
            # 只保留主要的 Noun 义项
            return any(s.pos() == 'n' for s in synsets)
        except: return True

    def detect_multiplicity(self, token):
        """
        基于语义分析推断 UML 的重数 (Multiplicity)
        """
        for child in token.children:
            if child.text.lower() in ["many", "multiple", "list", "set", "all", "collection"]: 
                return "1..*"
            if child.tag_ == "NNS": # 复数名词
                return "0..*"
        return "1"

    def process(self, text):
        if not self.nlp: return None
        
        self.classes = {}
        self.relationships = []
        doc = self.nlp(text)
        
        for token in doc:
            # --- Rule 1: Class Identification (基于 Ontology) ---
            if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "pobj", "nsubjpass"]:
                lemma = token.lemma_.lower()
                # 过滤黑名单 (根据论文设定)
                if lemma not in ["user", "data"]: 
                    if self.check_ontology(lemma):
                        c = token.lemma_.capitalize()
                        if c not in self.classes: 
                            self.classes[c] = {'attributes': set(), 'methods': set()}

            # --- Rule 2: Generalization (继承关系) ---
            if token.lemma_ == "be":
                subj = [c for c in token.children if c.dep_ == "nsubj"]
                attr = [c for c in token.children if c.dep_ == "attr"]
                if subj and attr:
                    c = subj[0].lemma_.capitalize()
                    p = attr[0].lemma_.capitalize()
                    if c in self.classes and p in self.classes: 
                        self.relationships.append((c, "Generalization", p, ""))
            
            # --- Rule 3: Composition/Aggregation (整体-部分) ---
            elif token.lemma_ in ["have", "contain", "include", "consist"]:
                owners = [c for c in token.children if c.dep_ == "nsubj"]
                objs = [c for c in token.children if c.dep_ == "dobj"]
                if owners and objs:
                    o = owners[0].lemma_.capitalize()
                    mult = self.detect_multiplicity(objs[0])
                    mlabel = mult if mult != "1" else ""
                    if o in self.classes:
                        obj_c = objs[0].lemma_.capitalize()
                        if obj_c in self.classes: 
                            # 识别为类之间的关系
                            self.relationships.append((o, "Composition", obj_c, mlabel))
                        else: 
                            # 降级为属性
                            self.classes[o]['attributes'].add(objs[0].text)

            # --- Rule 4: Association (常规关联) ---
            elif token.pos_ == "VERB" and token.lemma_ not in self.ignored_verbs:
                subjs = [c for c in token.children if c.dep_ == "nsubj"]
                if subjs:
                    s = subjs[0].lemma_.capitalize()
                    if s in self.classes:
                        self.classes[s]['methods'].add(token.lemma_)
                        dobjs = [c for c in token.children if c.dep_ == "dobj"]
                        if dobjs:
                            o = dobjs[0].lemma_.capitalize()
                            if o in self.classes and s != o: 
                                self.relationships.append((s, "Association", o, token.lemma_))

            # --- Rule 5: Passive Voice Handling (被动语态) ---
            # e.g., "Account is managed by Admin"
            if token.dep_ == "agent" and token.head.pos_ == "VERB":
                actual_agent = [c for c in token.children if c.dep_ == "pobj"] # Admin
                verb = token.head # managed
                passive_subj = [c for c in verb.children if c.dep_ == "nsubjpass"] # Account
                
                if actual_agent and passive_subj:
                    act = actual_agent[0].lemma_.capitalize()
                    rec = passive_subj[0].lemma_.capitalize()
                    
                    # 确保类存在
                    if act not in self.classes: self.classes[act] = {'attributes': set(), 'methods': set()}
                    if rec not in self.classes: self.classes[rec] = {'attributes': set(), 'methods': set()}
                    
                    self.classes[act]['methods'].add(verb.lemma_)
                    self.relationships.append((act, "Association", rec, verb.lemma_))

        return self.generate_graphviz()

    def generate_graphviz(self):
        """
        Phase 4: Rendering Engine (using Graphviz)
        完全符合论文中描述的 'Visual Mapping' 过程
        """
        dot = Digraph(comment='Thesis UML')
        dot.attr(rankdir='BT', splines='ortho', nodesep='0.8', ranksep='0.8')
        dot.attr('node', shape='record', style='filled', fillcolor='#FEFECE', fontname='Helvetica', fontsize='12')
        
        # 1. Nodes (Classes)
        for class_name, details in self.classes.items():
            attrs = r"\l".join([f"- {a}" for a in details['attributes']]) + r"\l" if details['attributes'] else ""
            methods = r"\l".join([f"+ {m}()" for m in details['methods']]) + r"\l" if details['methods'] else ""
            
            # 使用 Record Shape 模拟 UML 类图框
            label = f"{{ {class_name} | {attrs} | {methods} }}"
            dot.node(class_name, label=label)

        # 2. Edges (Relationships)
        unique_rels = set(self.relationships)
        for s, r_type, t, label in unique_rels:
            if r_type == "Generalization":
                dot.edge(s, t, arrowhead='onormal', label='') # 继承空心三角
            elif r_type == "Composition":
                dot.edge(s, t, dir='both', arrowtail='diamond', arrowhead='none', label=label) # 组合实心菱形
            else:
                dot.edge(s, t, arrowhead='vee', label=label) # 关联普通箭头

        return dot

# ==========================================
# 3. 界面交互 (Thesis Presentation UI)
# ==========================================
st.title("🎓 Automatic UML Generation System")
st.caption("Master's Thesis Project | NLP to UML Transformation Pipeline")

if nlp is None:
    st.error("⚠️ Model Loading Error")
    st.info("System could not load 'en_core_web_sm'. Please check requirements.txt.")
else:
    system = ThesisUMLSystem(nlp)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Input Requirements")
        txt = st.text_area("Natural Language Spec:", 
                          "The BankSystem contains many Accounts.\nAn Account is owned by a Customer.\nThe Administrator manages the System.", 
                          height=200)
        
        if st.button("Generate Diagram", type="primary"):
            with st.spinner("Analyzing semantics..."):
                try:
                    graph = system.process(txt)
                    st.session_state['graph_dot'] = graph
                    st.session_state['generated_classes'] = system.classes
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")

        # Evaluation Section
        st.markdown("---")
        st.subheader("3. Validation (F1-Score)")
        gt = st.text_input("Ground Truth Classes (comma-separated):", "BankSystem, Account, Customer, Administrator")
        
        if st.button("Calculate Metrics"):
            if st.session_state['generated_classes']:
                # F1 Calculation Logic
                exp = set([x.strip().lower() for x in gt.split(",") if x.strip()])
                det = set([x.lower() for x in st.session_state['generated_classes'].keys()])
                
                tp = len(exp.intersection(det))
                fp = len(det - exp)
                fn = len(exp - det)
                
                p = tp/(tp+fp) if (tp+fp)>0 else 0
                r = tp/(tp+fn) if (tp+fn)>0 else 0
                f1 = 2*(p*r)/(p+r) if (p+r)>0 else 0
                
                st.metric("F1-Score", f"{f1:.2f}")
                st.text(f"Precision: {p:.2f} | Recall: {r:.2f}")
            else:
                st.warning("Generate a diagram first.")

    with col2:
        st.subheader("2. Generated Model")
        if st.session_state['graph_dot']:
            st.graphviz_chart(st.session_state['graph_dot'])
            with st.expander("Show DOT Source (for verification)"):
                st.code(st.session_state['graph_dot'].source)
        else:
            st.info("Waiting for input...")