#!/usr/bin/env python3
"""
Quantum CPI Prediction Portal
Uses the trained quantum model from the notebook for inference only.
"""

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
import pennylane as qml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import BertModel, BertTokenizer

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Q-CPID",
    page_icon="🧬",
    layout="wide",
)

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top, rgba(79, 70, 229, 0.32), transparent 35%),
                        linear-gradient(135deg, #050816 0%, #0b1022 40%, #111827 100%);
                color: #ffffff;
        }
        .stApp {
            background: transparent;
                color: #ffffff;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        header[data-testid="stHeader"] [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none;
        }
        .stMarkdown, .stMarkdown p, .stText, label,
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
                color: #ffffff !important;
        }
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
                color: #ffffff !important;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }
        .main-header {
            font-size: 2.8rem;
            font-weight: 800;
            text-align: center;
            background: linear-gradient(90deg, #8b5cf6 0%, #22d3ee 50%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.8rem;
            letter-spacing: 0.04em;
        }
        .sub-header {
            font-size: 1.4rem;
            font-weight: 700;
                color: #ffffff;
            margin-bottom: 0.9rem;
        }
        .hero-panel {
            background: rgba(17, 24, 39, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 20px;
            box-shadow: 0 0 35px rgba(59, 130, 246, 0.25);
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            background-image: linear-gradient(rgba(15, 23, 42, 0.65), rgba(15, 23, 42, 0.72)),
                              url('https://www.dropbox.com/scl/fi/uhvdzarqh2lauoqsatdvr/banner.png?rlkey=zgwmvwm186tn3ovpwyskt1fcx&st=1mhq6c6k&dl=1');
            background-size: cover;
            background-position: center;
        }
        .metric-card, .glass-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.45);
            padding: 1rem 1.05rem;
            border-radius: 16px;
                color: #ffffff;
        }
        .prediction-result {
            background: linear-gradient(135deg, rgba(99,102,241,0.9), rgba(168,85,247,0.9));
            color: white;
            padding: 1.5rem 1.2rem;
            border-radius: 18px;
            text-align: center;
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 0 25px rgba(168, 85, 247, 0.35);
        }
        .stButton > button {
            background: linear-gradient(90deg, #7c3aed 0%, #06b6d4 100%);
            color: white;
            border: none;
            border-radius: 999px;
            font-weight: 700;
            padding: 0.7rem 1.4rem;
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.38);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(59, 130, 246, 0.5);
        }
        .stTextInput > div > div > input,
        .stTextArea > div > textarea,
        .stSelectbox > div > div,
        .stMultiSelect > div > div,
        .stNumberInput > div > div > input {
            background: rgba(15, 23, 42, 0.82);
            color: #ffffff !important;
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 12px;
        }
        .stSelectbox [data-baseweb="select"] *,
        .stMultiSelect [data-baseweb="select"] *,
        .stRadio label,
        .stRadio label p,
        .stRadio label span,
        .stCheckbox label,
        .stCheckbox label p,
        .stCheckbox label span {
                color: #ffffff !important;
        }
        .stCheckbox > label {
                color: #ffffff;
        }
        div[data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.8);
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        .stAlert {
            background: rgba(59, 130, 246, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.25);
                color: #ffffff;
        }
        .stAlert *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        .prediction-result h3,
        .prediction-result p,
        .prediction-result strong {
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

AA = 'ACDEFGHIKLMNPQRSTVWY'


def load_dataset_protein_presets():
    """Load the six unique protein sequences found in the breast-cancer dataset."""
    aliases = {
        "MSAIQAAWPSGTECIAKYNFHGTAEQDLPFCKGDVLTIVAVTKDPNWYKAKNKVGREGIIPANYVQKREGVKAGTKLSLMPWFHGKITREQAERLLYPPETGLFLVRESTNYPGDYTLCVSCDGKVEHYRIMYHASKLSIDEEVYFENLMQLVEHYTSDADGLCTRLIKPKVMEGTVAAQDEFYRSGWALNMKELKLLQTIGKGEFGDVMLGDYRGNKVAVKCIKNDATAQAFLAEASVMTQLRHSNLVQLLGVIVEEKGGLYIVTEYMAKGSLVDYLRSRGRSVLGGDCLLKFSLDVCEAMEYLEGNNFVHRDLAARNVLVSEDNVAKVSDFGLTKEASSTQDTGKLPVKWTAPEALREKKFSTKSDVWSFGILLWEIYSFGRVPYPRIPLKDVVPRVEKGYKMDAPDGCPPAVYEVMKNCWHLDAAMRPSFLQLREQLEHIKTHELHL": "CSK (P41240)",
        "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQEPESGKVVQEGFLREPGPPGLSHQLMSGMPGAPLLPEGPREATRQPSGTGPEDTEGGRHAPELLKHQLLGDLHQEGPPLKGAGGKERPGSKEEVDEDRDVDESSPQDSPPSKASPAQDGRPPQTAAREATSIPGFPAEGAIPLPVDFLSKVSTEIPASEPDGPSVGRAKGQDAPLEFTFHVEITPNVQKEQAHSEEHLGRAAFPGAPGEGPEARGPSLGEDTKEADLPEPSEKQPAAAPRGKPVSRVPQLKARMVSKSKDGTGSDDKKAKTSTRSSAKTLKNRPCLSPKHPTPGSSDPLIQPSSPAVCPEPPSSPKYVSSVTSRTGSSGAKEMKLKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL": "MAPT (P10636)",
        "MCNTNMSVPTDGAVTTSQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSENRCHLEGGSDQKDLVQELQEEKPSSSHLVSRPSTSSRRRAISETEENSDELSGERQRKRHKSDSISLSFDESLALCVIREICCERSSSSESTGTPSNPDLDAGVSEHSGDWLDQDSVSDQFSVEFEVESLDSEDYSLSEEGQELSDEDDEVYQVTVYQAGESDTDSFEEDPEISLADYWKCTSCNEMNPPLPSHCNRCWALRENWLPEDKGKDKGEISEKAKLENSTQAEEGFDVPDCKKTIVNDSRESCVEENDDKITQASQSQESEDYSQPSTSSSIIYSSQEDVKEFEREETQDKEESVESSLPLNAIEPCVICQGRPKNGCIVHGKTGHLMACFTCAKKLKKRNKPCPVCRQPIQMIVLTYFP": "MDM2 (Q00987)",
        "MGGDLVLGLGALRRRKRLLEQEKSLAGWALVLAGTGIGLMVLHAEMLWFGGCSWALYLFLVKCTISISTFLLLCLIVAFHAKEVQLFMTDNGLRDWRVALTGRQAAQIVLELVVCGLHPAPVRGPPCVQDLGAPLTSPQPWPGFLGQGEALLSLAMLLRLYLVPRAVLLRSGVLLNASYRSIGALNQVRFRHWFVAKLYMNTHPGRLLLGLTLGLWLTTAWVLSVAERQAVNATGHLSDTLWLIPITFLTIGYGDVVPGTMWGKIVCLCTGVMGVCCTALLVAVVARKLEFNKAEKHVHNFMMDIQYTKEMKESAARVLQEAWMFYKHTRRKESHAARRHQRKLLAAINAFRQVRLKHRKLREQVNSMVDISKMHMILYDLQQNLSSSHRALEKQIDTLAGKLDALTELLSTALGPRQLPEPSQQSK": "KCNN4 (O15554)",
        "MAVQGSQRRLLGSLNSTPTAIPQLGLAANQTGARCLEVSISDGLFLSLGLVSLVENALVVATIAKNRNLHSPMYCFICCLALSDLLVSGSNVLETAVILLLEAGALVARAAVLQQLDNVIDVITCSSMLSSLCFLGAIAVDRYISIFYALRYHSIVTLPRARRAVAAIWVASVVFSTLFIAYYDHVAVLLCLVVFFLAMLVLMAVLYVHMLARACQHAQGIARLHKRQRPVHQGFGLKGAVTLTILLGIFFLCWGPFFLHLTLIVLCPEHPTCGCIFKNFNLFLALIICNAIIDPLIYAFHSQELRRTLKEVLTCSW": "MC1R (Q01726)",
        "MREIVHIQAGQCGNQIGAKFWEVISDEHGIDPTGTYHGDSDLQLDRISVYYNEATGGKYVPRAILVDLEPGTMDSVRSGPFGQIFRPDNFVFGQSGAGNNWAKGHYTEGAELVDSVLDVVRKEAESCDCLQGFQLTHSLGGGTGSGMGTLLISKIREEYPDRIMNTFSVVPSPKVSDTVVEPYNATLSVHQLVENTDETYCIDNEALYDICFRTLKLTTPTYGDLNHLVSATMSGVTTCLRFPGQLNADLRKLAVNMVPFPRLHFFMPGFAPLTSRGSQQYRALTVPELTQQVFDAKNMMAACDPRHGRYLTVAAVFRGRMSMKEVDEQMLNVQNKNSSYFVEWIPNNVKTAVCDIPPRGLKMAVTFIGNSTAIQELFKRISEQFTAMFRRKAFLHWYTGEGMDEMEFTEAESNMNDLVSEYQQYQDATAEEEEDFGEEAEEEA": "TUBB (P07437)",
    }

    unique_sequences = list(aliases)
    if len(unique_sequences) != 6:
        raise ValueError(f"Expected 6 unique protein sequences, found {len(unique_sequences)}")

    protein_presets = []
    for index, seq in enumerate(unique_sequences[:6], start=1):
        gene_name = aliases.get(seq, f"Protein-{index:02d}")
        protein_presets.append({"gene_name": gene_name, "sequence": seq})

    return protein_presets


DATASET_SMILES_SAMPLES = (
    "N[C@@H](Cc1ccc(Br)cc1)C(=O)NO",
    "FC(F)(F)C(=O)c1ccncc1NC(=O)c1ccnc(NC(=O)C2CC2)c1",
    "CC(C)(C)OC(=O)Nc1ccc(cc1)-c1cn(CCC[C@H](NC(=O)OCC2c3ccccc3-c3ccccc23)C(O)=O)nn1",
    "CC[C@@H](CS(=O)(=O)CC1(C)COC1)N1[C@@H]([C@H](C[C@](C)(CC(O)=O)C1=O)c1cccc(Cl)c1)c1ccc(Cl)cc1",
    "COc1ncc(-c2cc3C(=O)N([C@H](c3n2C(C)C)c2ccc(cc2)C#N)c2cc(Cl)cn(C)c2=O)c(OC)n1",
    "Cc1nc2N(Cc3cccc(c3)C(F)(F)F)C(=O)CSc2s1",
    "COc1ccccc1-c1cc2C(=O)N(C(c2n1C(C)C)c1ccc(Cl)cc1)c1cc(Cl)c(=O)n(C)c1",
    "CCC1(CC)N[C@H]([C@H](c2cccc(Cl)c2F)[C@]11C(=O)Nc2cc(Cl)ccc12)C(=O)N[C@H]1CC[C@H](O)CC1",
)


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = []
    for atom in mol.GetAtoms():
        x.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetHybridization())
        ])
    x = torch.tensor(x, dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def morgan_fp(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return torch.tensor(arr, dtype=torch.float)


def protein_aac(seq):
    seq = seq.upper()
    if len(seq) == 0:
        return torch.zeros((len(AA),), dtype=torch.float)
    return torch.tensor([seq.count(a) / len(seq) for a in AA], dtype=torch.float)


def generate_smiles_from_structure(structure_type, chain_length, is_branch,
                                   add_oh, add_nh2, add_cooh, add_ch3, add_cl, add_br, add_f, add_no2, add_cn):
    base_structures = {
        "Simple Chain": "C" * chain_length,
        "Benzene Ring": "c1ccccc1",
        "Pyridine": "c1ccncc1",
        "Pyrimidine": "c1cncnc1",
        "Imidazole": "c1c[nH]nc1",
        "Custom": "C" * chain_length,
    }

    smiles = base_structures.get(structure_type, "C" * chain_length)
    functional_groups = []

    if add_oh:
        functional_groups.append("O")
    if add_nh2:
        functional_groups.append("N")
    if add_cooh:
        functional_groups.append("C(=O)O")
    if add_ch3:
        functional_groups.append("C")
    if add_cl:
        functional_groups.append("Cl")
    if add_br:
        functional_groups.append("Br")
    if add_f:
        functional_groups.append("F")
    if add_no2:
        functional_groups.append("N(=O)=O")
    if add_cn:
        functional_groups.append("C#N")

    if functional_groups and structure_type != "Simple Chain":
        smiles += functional_groups[0]
    elif functional_groups:
        for fg in functional_groups[:2]:
            smiles += fg

    if is_branch and chain_length > 2:
        branch_pos = min(2, len(smiles) - 1)
        smiles = smiles[:branch_pos] + "(C)" + smiles[branch_pos:]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return smiles
        return "C" * chain_length
    except Exception:
        return "C" * chain_length


class CompressionLayer(nn.Module):
    def __init__(self, input_dim=2196, output_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.tanh(x)


class QuantumCircuitSimplified(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        n_params = n_layers * n_qubits * 3
        self.params = nn.Parameter(torch.randn(n_params) * 0.1)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")

    def quantum_circuit(self, inputs, weights):
        weights = weights.reshape(self.n_layers, self.n_qubits, 3)
        for i in range(min(self.n_qubits, len(inputs))):
            qml.RY(inputs[i] * np.pi, wires=i)

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.RX(weights[layer, qubit, 0], wires=qubit)
                qml.RY(weights[layer, qubit, 1], wires=qubit)
                qml.RZ(weights[layer, qubit, 2], wires=qubit)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            output = self.qnode(x[i], self.params)
            output = torch.stack(output)
            outputs.append(output)
        return torch.stack(outputs).float()


class VQCClassifier(nn.Module):
    def __init__(self, n_qubits=8):
        super().__init__()
        self.compression = CompressionLayer(input_dim=2196, output_dim=n_qubits)
        self.vqc = QuantumCircuitSimplified(n_qubits=n_qubits, n_layers=2)
        self.post_quantum = nn.Sequential(nn.Linear(n_qubits, 1))

    def forward(self, features_batch):
        compressed = self.compression(features_batch)
        quantum_out = self.vqc(compressed)
        output = self.post_quantum(quantum_out.unsqueeze(1))
        return output.squeeze()


class CompoundGNN(nn.Module):
    def __init__(self, node_dim=4, hidden=128):
        super().__init__()
        self.conv1 = GCNConv(node_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)


class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, seqs):
        seqs = [" ".join(list(s)) for s in seqs]
        inputs = self.tokenizer(
            seqs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :]


class HybridCPI(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnn = CompoundGNN()
        self.protein = ProteinEncoder()

    def forward(self, batch):
        gnn_emb = self.gnn(batch.x, batch.edge_index, batch.batch)
        fp = batch.fp
        aac = batch.aac
        prot_emb = self.protein(batch.protein_seq)
        features = torch.cat([gnn_emb, fp, prot_emb, aac], dim=1)
        return features


class FullModel(nn.Module):
    def __init__(self, n_qubits=8):
        super().__init__()
        self.feature_extractor = HybridCPI()
        self.vqc = VQCClassifier(n_qubits=n_qubits)

    def forward(self, batch):
        features = self.feature_extractor(batch)
        return self.vqc(features)


def build_inference_batch(smiles, protein_seq):
    data = smiles_to_graph(smiles)
    data.fp = morgan_fp(smiles).unsqueeze(0)
    data.aac = protein_aac(protein_seq).unsqueeze(0)
    data.protein_seq = [protein_seq]
    data.y = torch.tensor([0.0], dtype=torch.float)
    return DataLoader([data], batch_size=1, shuffle=False)


@st.cache_resource
def load_quantum_model(model_path="best_model_run1.pt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    model = FullModel(n_qubits=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_interaction(model, loader):
    batch = next(iter(loader))
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
        prob = float(probs.cpu().item())
        pred = int(prob >= 0.5)
    return prob, pred


def main():
    st.markdown(
        """
        <div class="hero-panel">
            <h1 class="main-header">🧬 Q-CPID</h1>
            <p style="text-align:center; color:#ffffff; margin:0; font-size:1.05rem;">
                Quantum-Driven Compound-Protein Interaction Prediction for Accelerating Breast Cancer Drug Discovery
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "smiles_input" not in st.session_state:
        st.session_state.smiles_input = ""
    if "protein_input" not in st.session_state:
        st.session_state.protein_input = ""

    protein_presets = load_dataset_protein_presets()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">🧪 Compound Input</h2>', unsafe_allow_html=True)
        input_method = st.radio("Input Method", ["SMILES String", "Construct Molecule"], horizontal=True)

        if input_method == "SMILES String":
            if DATASET_SMILES_SAMPLES:
                selected_sample = st.selectbox(
                    "Try a dataset sample",
                    options=[""] + list(DATASET_SMILES_SAMPLES),
                    format_func=lambda value: (
                        "Choose a sample..."
                        if not value
                        else f"{value[:42]}..." if len(value) > 42 else value
                    ),
                )
                if selected_sample:
                    st.session_state.smiles_input = selected_sample
                    st.session_state.smiles_text_area = selected_sample

            smiles_input = st.text_area(
                "Enter SMILES String:",
                value=st.session_state.smiles_input,
                placeholder="e.g., CC(C)OC(=O)N1CCC(N2C(=O)c3ccccc3C2=O)CC1",
                height=120,
                key="smiles_text_area"
            )
            if smiles_input != st.session_state.smiles_input:
                st.session_state.smiles_input = smiles_input
        else:
            with st.container():
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                structure_type = st.selectbox(
                    "Select Base Structure:",
                    ["Simple Chain", "Benzene Ring", "Pyridine", "Pyrimidine", "Imidazole", "Custom"]
                )
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    add_oh = st.checkbox("Hydroxyl (-OH)")
                    add_nh2 = st.checkbox("Amino (-NH₂)")
                    add_cooh = st.checkbox("Carboxyl (-COOH)")
                with col_b:
                    add_ch3 = st.checkbox("Methyl (-CH₃)")
                    add_cl = st.checkbox("Chlorine (-Cl)")
                    add_br = st.checkbox("Bromine (-Br)")
                with col_c:
                    add_f = st.checkbox("Fluorine (-F)")
                    add_no2 = st.checkbox("Nitro (-NO₂)")
                    add_cn = st.checkbox("Cyano (-CN)")

                chain_length = st.slider("Carbon Chain Length:", 1, 10, 3)
                is_branch = st.checkbox("Add Branching")

                if st.button("🔬 Generate SMILES"):
                    generated_smiles = generate_smiles_from_structure(
                        structure_type,
                        chain_length,
                        is_branch,
                        add_oh,
                        add_nh2,
                        add_cooh,
                        add_ch3,
                        add_cl,
                        add_br,
                        add_f,
                        add_no2,
                        add_cn,
                    )
                    st.session_state.smiles_input = generated_smiles
                    st.success(f"Generated SMILES: {generated_smiles}")
                st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.smiles_input:
            try:
                mol = Chem.MolFromSmiles(st.session_state.smiles_input)
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(360, 260))
                    st.image(img, caption="Compound graph view")#, use_column_width=False)
                    st.code(st.session_state.smiles_input, language="text")
                else:
                    st.error("Invalid SMILES string.")
            except Exception as exc:
                st.error(f"Error parsing SMILES: {exc}")

    with col2:
        st.markdown('<h2 class="sub-header">🧬 Protein Input</h2>', unsafe_allow_html=True)
        st.caption("Using six embedded protein sequences from the breast-cancer dataset")

        protein_source = st.radio("Protein Source", ["Dataset Preset", "Custom Sequence"], horizontal=True)

        if protein_source == "Dataset Preset":
            preset = st.selectbox(
                "Select protein preset",
                options=protein_presets,
                format_func=lambda item: f"{item['gene_name']} ({len(item['sequence'])} aa)",
            )
            st.session_state.protein_input = preset["sequence"]
            st.info(f"Selected gene alias: {preset['gene_name']}")
            st.code(preset["sequence"][:80] + "...", language="text")
        else:
            protein_input = st.text_area(
                "Enter Protein Sequence:",
                value=st.session_state.protein_input,
                placeholder="Enter amino acid sequence (e.g., MKWVTFISLLFLFSSAYSRGVF...)",
                height=180,
                key="protein_text_area"
            )
            st.session_state.protein_input = protein_input

        protein_sequence = st.session_state.protein_input.strip()
        if protein_sequence:
            st.info(f"Sequence Length: {len(protein_sequence)} amino acids")
            seq = protein_sequence.upper()
            composition = {aa: seq.count(aa) for aa in AA if seq.count(aa) > 0}
            if composition:
                comp_df = pd.DataFrame(list(composition.items()), columns=["Amino Acid", "Count"])
                st.dataframe(comp_df, use_container_width=True)

    st.markdown("---")

    if st.button("🚀 Predict Interaction", use_container_width=True):
        smiles_value = st.session_state.smiles_input.strip()
        protein_value = st.session_state.protein_input.strip()

        if not smiles_value or not protein_value:
            st.error("Please provide both compound SMILES and protein sequence.")
            return

        try:
            mol = Chem.MolFromSmiles(smiles_value)
            if mol is None:
                st.error("Invalid SMILES string.")
                return
        except Exception:
            st.error("Invalid SMILES string.")
            return

        if len(protein_value) < 10:
            st.error("Protein sequence is too short.")
            return

        with st.spinner("Loading quantum model and predicting..."):
            try:
                model = load_quantum_model()
                loader = build_inference_batch(smiles_value, protein_value)
                prob, pred = predict_interaction(model, loader)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        prediction_text = "Interaction" if pred == 1 else "No Interaction"
        confidence = "High" if prob >= 0.75 or prob <= 0.25 else "Medium" if prob >= 0.6 or prob <= 0.4 else "Low"

        st.markdown('<h2 class="sub-header">📊 Prediction Results</h2>', unsafe_allow_html=True)
        metric_col1, metric_col2, metric_col3 = st.columns([1, 1, 1])
        with metric_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Interaction Probability", f"{prob:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with metric_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Prediction", prediction_text)
            st.markdown('</div>', unsafe_allow_html=True)
        with metric_col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", confidence)
            st.markdown('</div>', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(
                f"""
                <div class="prediction-result">
                    <h3>🎯 Interaction Detected</h3>
                    <p>Estimated probability: <strong>{prob:.1%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-result" style="background: linear-gradient(135deg, #f472b6 0%, #fb7185 100%);">
                    <h3>❌ No Interaction Detected</h3>
                    <p>Estimated probability: <strong>{prob:.1%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        try:
            import plotly.graph_objects as go

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Interaction Probability (%)", "font": {"color": "#ffffff"}},
                    number={"font": {"color": "#ffffff"}},
                    delta={"reference": 50, "font": {"color": "#ffffff"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"color": "#ffffff"}},
                        "bar": {"color": "#8b5cf6"},
                        "steps": [
                            {"range": [0, 25], "color": "#1f2937"},
                            {"range": [25, 50], "color": "#374151"},
                            {"range": [50, 75], "color": "#4f46e5"},
                            {"range": [75, 100], "color": "#22d3ee"},
                        ],
                        "threshold": {"line": {"color": "#fda4af", "width": 4}, "thickness": 0.75, "value": 90},
                    },
                )
            )
            fig.update_layout(
                height=360,
                margin={"t": 20, "b": 20, "l": 20, "r": 20},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#ffffff"},
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #ffffff;'>
            <p>🧬 Q-CPID</p>
            <p>Developed by Bhomic Kaushik</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
