import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import networkx as nx

#NOTA
#Esto es solo un ejemplo teorico poco practico y muy basico para entender los componentes a nivel bajo de un LLM


# Codificación Posicional: Añade información sobre la posición de cada token en la secuencia
class EncodingPosicional(nn.Module):
    def __init__(self, dim_modelo, longitud_max=5000):
        super().__init__()
        posicion = torch.arange(longitud_max).unsqueeze(1)
        termino_div = torch.exp(torch.arange(0, dim_modelo, 2) * (-math.log(10000.0) / dim_modelo))
        pe = torch.zeros(longitud_max, 1, dim_modelo)
        pe[:, 0, 0::2] = torch.sin(posicion * termino_div)
        pe[:, 0, 1::2] = torch.cos(posicion * termino_div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# Modelo Transformer: Arquitectura principal del LLM
class ModeloTransformer(nn.Module):
    def __init__(self, num_tokens, dim_modelo, num_cabezas, dim_oculta, num_capas, dropout=0.5):
        super().__init__()
        self.tipo_modelo = 'Transformer'
        self.codificador_posicional = EncodingPosicional(dim_modelo)
        # Capa de codificación del Transformer
        capa_encoding = nn.TransformerEncoderLayer(dim_modelo, num_cabezas, dim_oculta, dropout, batch_first=True)
        # Apilamiento de capas encodings
        self.transformer_encoder = nn.TransformerEncoder(capa_encoding, num_capas)
        # Embedding: Convierte tokens en vectores 
        self.embedding = nn.Embedding(num_tokens, dim_modelo)
        self.dim_modelo = dim_modelo
        # Capa de salida: Proyecta las representaciones a probabilidades de tokens
        self.output = nn.Linear(dim_modelo, num_tokens)

    def forward(self, src, mascara_src):
        # Proceso de forward del modelo
        src = self.embedding(src) * math.sqrt(self.dim_modelo)
        src = self.codificador_posicional(src)
        # Atención multi-cabeza y feed-forward en el codificador
        salida = self.transformer_encoder(src, mascara_src)
        salida = self.output(salida)
        return salida

# El número de parámetros en el modelo se define por la arquitectura y los hiperparámetros.
# Para ver el número de parámetros, podemos usar:
#
# modelo = ModeloTransformer(num_tokens, dim_modelo, num_cabezas, dim_oculta, num_capas)
# total_params = sum(p.numel() for p in modelo.parameters())
# print(f"Número total de parámetros: {total_params}")
#
# También podemos ver un desglose detallado de los parámetros por capa:
#
# for name, param in modelo.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()}")

# Genera una máscara para evitar que el modelo "vea" tokens futuros
def gen_mascara_subsecuente(tamano):
    mascara = (torch.triu(torch.ones(tamano, tamano)) == 1).transpose(0, 1)
    mascara = mascara.float().masked_fill(mascara == 0, float('-inf')).masked_fill(mascara == 1, float(0.0))
    return mascara

# Dataset personalizado para refranes
class DatasetRefranes(Dataset):
    def __init__(self, refranes, longitud_seq):
        self.refranes = refranes
        self.longitud_seq = longitud_seq
        self.caracteres = sorted(list(set(''.join(refranes))))
        self.char_a_idx = {ch: i for i, ch in enumerate(self.caracteres)}
        self.idx_a_char = {i: ch for i, ch in enumerate(self.caracteres)}
        self.datos = self.preparar_datos()

    def preparar_datos(self):
        datos = []
        for refran in self.refranes:
            codificado = [self.char_a_idx[ch] for ch in refran]
            for i in range(0, len(codificado) - self.longitud_seq):
                datos.append((codificado[i:i+self.longitud_seq], codificado[i+1:i+self.longitud_seq+1]))
        return datos

    def __len__(self):
        return len(self.datos)

    def __getitem__(self, idx):
        return torch.tensor(self.datos[idx][0]), torch.tensor(self.datos[idx][1])

# Función de entrenamiento del modelo
def entrenar_modelo(modelo, cargador_entrenamiento, criterio, optimizador, dispositivo):
    modelo.train()
    loss_total = 0
    for src, tgt in cargador_entrenamiento:
        src, tgt = src.to(dispositivo), tgt.to(dispositivo)
        mascara_src = gen_mascara_subsecuente(src.size(1)).to(dispositivo)
        
        optimizador.zero_grad()
        salida = modelo(src, mascara_src)
        loss = criterio(salida.view(-1, salida.size(-1)), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 0.5)
        optimizador.step()
        
        loss_total += loss.item()
    return loss_total / len(cargador_entrenamiento)

# Función para gen texto new
def gen_texto(modelo, dataset, texto_inicial, longitud_max=100):
    modelo.eval()
    dispositivo = next(modelo.parameters()).device
    seq_entrada = torch.tensor([dataset.char_a_idx[ch] for ch in texto_inicial]).unsqueeze(0).to(dispositivo)
    texto_generado = texto_inicial


    with open('parameters.txt', 'w') as f:
        for name, param in modelo.named_parameters():
            f.write(f"{name}: {param.data}\n")


    with open('table.txt', 'w') as f:
        f.write("Posición\tCarácter\tProbabilidad\n")

    with torch.no_grad():
        for pos in range(longitud_max - len(texto_inicial)):
            mascara_src = gen_mascara_subsecuente(seq_entrada.size(1)).to(dispositivo)
            salida = modelo(seq_entrada, mascara_src)
            probabilidades = torch.softmax(salida[:, -1, :], dim=-1)
            idx_siguiente_char = torch.multinomial(probabilidades, 1).item()
            siguiente_char = dataset.idx_a_char[idx_siguiente_char]
            texto_generado += siguiente_char
            seq_entrada = torch.cat([seq_entrada, torch.tensor([[idx_siguiente_char]]).to(dispositivo)], dim=1)
            
           
            top_k = 5
            top_probs, top_indices = torch.topk(probabilidades, top_k)
            print("\nTokens más probables:")
            with open('table.txt', 'a') as f:
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    char = dataset.idx_a_char[idx.item()]
                    probability = prob.item()
                    print(f"{char}: {probability:.4f}")
                    f.write(f"{pos+1}\t{char}\t{probability:.4f}\n")
            
            if siguiente_char == '.':
                break

    return texto_generado

# Configuración y entrenamiento
refranes = [
    "Quien madruga con el sol, cosecha éxito y honor.",
    "En la paciencia está el sabor, en la prisa el error.",
    "Más vale un amigo cercano que cien parientes lejanos.",
    "El que ríe al caer, se levanta sin perder.",
    "No es más rico quien más tiene, sino quien menos necesita.",
    "La palabra amable abre puertas de acero.",
    "El árbol de la sabiduría crece regado con humildad.",
    "Quien siembra cortesía, cosecha respeto.",
    "En la diversidad de opiniones, florece la verdad.",
    "El silencio es oro, pero la palabra oportuna es diamante.",
    "Más vale prevenir hoy que lamentar mañana.",
    "La constancia mueve montañas, la pereza solo excusas.",
    "No hay atajo sin trabajo, ni éxito sin esfuerzo.",
    "La gratitud es la memoria del corazón.",
    "Quien cultiva la paz, recoge armonía.",
    "El sabio escucha más y habla menos.",
    "La honestidad es la mejor política.",
    "Una mente abierta es un tesoro invaluable.",
    "El respeto se gana, no se exige.",
    "La perseverancia es la clave del éxito.",
    "El conocimiento es poder, la sabiduría es libertad.",
    "La verdadera riqueza está en los amigos.",
    "Quien perdona, libera su alma.",
    "La humildad es la base de todas las virtudes.",
    "El tiempo es oro, no lo desperdicies.",
    "La unión hace la fuerza, la división la debilidad.",
    "Más vale tarde que nunca.",
    "La vida es corta, pero el arte es largo.",
    "El que busca, encuentra.",
    "No dejes para mañana lo que puedas hacer hoy.",
    "La esperanza es el pan de los pobres.",
    "Quien mucho abarca, poco aprieta.",
    "La paciencia es amarga, pero su fruto es dulce.",
    "El buen juez por su casa empieza.",
    "Más sabe el diablo por viejo que por diablo.",
    "Quien a buen árbol se arrima, buena sombra le cobija.",
    "No por mucho madrugar amanece más temprano.",
    "Más vale pájaro en mano que ciento volando.",
    "A caballo regalado no se le mira el diente.",
    "En casa del herrero, cuchillo de palo.",
    "Dime con quién andas y te diré quién eres.",
    "No todo lo que brilla es oro.",
    "Quien calla, otorga.",
    "A palabras necias, oídos sordos.",
    "El que ríe último, ríe mejor.",
    "Más vale solo que mal acompañado.",
    "La curiosidad mató al gato.",
    "El hábito no hace al monje.",
    "Ojos que no ven, corazón que no siente.",
    "Cuando el río suena, piedras lleva.",
    "No hay mal que por bien no venga.",
    "A Dios rogando y con el mazo dando.",
    "El que no arriesga, no gana.",
    "Cada oveja con su pareja.",
    "De tal palo, tal astilla.",
    "El que a hierro mata, a hierro muere.",
    "En boca cerrada no entran moscas.",
    "Haz bien y no mires a quién.",
    "La ocasión hace al ladrón.",
    "Más vale maña que fuerza.",
    "No hay mal que cien años dure.",
    "Perro ladrador, poco mordedor.",
    "Quien bien te quiere te hará llorar.",
    "Tanto va el cántaro a la fuente que al final se rompe.",
    "Una golondrina no hace verano.",
    "Zapatero a tus zapatos.",
    "A falta de pan, buenas son tortas.",
    "Cría fama y échate a dormir.",
    "Del dicho al hecho hay mucho trecho.",
    "El que espera, desespera.",
    "Gota a gota, el agua se agota.",
    "La fe mueve montañas.",
    "Más vale prevenir que curar.",
    "No hay rosa sin espinas.",
    "Quien siembra vientos, recoge tempestades.",
    "Todo lo que sube tiene que bajar.",
    "A buen entendedor, pocas palabras bastan.",
    "Cada loco con su tema.",
    "El fin justifica los medios.",
    "La ropa sucia se lava en casa.",
    "Más vale un 'toma' que dos 'te daré'.",
    "No por mucho repetir una mentira se convierte en verdad.",
    "Quien tiene boca se equivoca.",
    "Una manzana podrida pudre el cesto.",
    "Al pan, pan, y al vino, vino.",
    "Camarón que se duerme se lo lleva la corriente.",
    "El mundo es un pañuelo.",
    "La suerte de la fea, la bonita la desea.",
    "Nadie es profeta en su tierra.",
    "Ojo por ojo, diente por diente.",
    "Quien mal anda, mal acaba.",
    "Una imagen vale más que mil palabras.",
    "A lo hecho, pecho.",
    "Cada uno sabe dónde le aprieta el zapato.",
    "El que parte y reparte se queda con la mejor parte.",
    "La unión hace la fuerza.",
    "No hay peor ciego que el que no quiere ver.",
    "Piensa mal y acertarás.",
    "Quien ríe el último, ríe mejor.",
    "Vísteme despacio que tengo prisa.",
]
refranes = refranes * 5
len_seq = 16
batch_size = 20
dataset = DatasetRefranes(refranes, len_seq)
cargador_entrenamiento = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tamano_vocabulario = len(dataset.caracteres)
dim_modelo = 128
num_cabezas = 4
dim_oculta = 256
num_capas = 2
tasa_dropout = 0.2

modelo = ModeloTransformer(tamano_vocabulario, dim_modelo, num_cabezas, dim_oculta, num_capas, tasa_dropout).to(dispositivo)
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# Mostrar el número total de parámetros del modelo
total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print(f"Número total de parámetros del modelo: {total_params}")

# Función para crear un fotograma de la animación
def crear_fotograma(epoch, modelo):
    num_params = sum(1 for name, _ in modelo.named_parameters() if _[1].requires_grad)
    rows = (num_params + 2) // 3  # Ensure we have enough rows
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    fig.suptitle(f"Estado de la red neuronal - Época {epoch+1}", fontsize=16)
    
    axs = axs.flatten() if num_params > 3 else [axs]  # Flatten if more than one row
    
    for i, (nombre, param) in enumerate(modelo.named_parameters()):
        if param.requires_grad:
            ax = axs[i]
            valores = param.detach().cpu().numpy()
            if len(valores.shape) == 2:
                im = ax.imshow(valores, cmap='viridis', aspect='auto')
                fig.colorbar(im, ax=ax)
            elif len(valores.shape) == 1:
                ax.plot(valores)
            ax.set_title(nombre)
            ax.set_xlabel("Dimensión 1")
            ax.set_ylabel("Dimensión 2" if len(valores.shape) == 2 else "Valor")
    
    # Hide any unused subplots
    for j in range(i+1, len(axs)):
        axs[j].axis('off')
    
    plt.tight_layout()
    return fig

# Lista para almacenar los fotogramas
fotogramas = []

num_epochs = 100
for epoch in range(num_epochs):
    loss = entrenar_modelo(modelo, cargador_entrenamiento, criterio, optimizador, dispositivo)
    if (epoch + 1) % 10 == 0:
        print(f'Época {epoch+1}/{num_epochs}, Pérdida: {loss:.4f}')
    
    # Crear y guardar un fotograma para la animación en cada época
        fig = crear_fotograma(epoch, modelo)
        plt.savefig(f'epoch_{epoch+1}.png')
        fotogramas.append(fig)
        plt.close(fig)


print("\nGenerando nuevos refranes:")
for _ in range(5):
    texto_inicial = "No hay "
    try:
        new_refran = gen_texto(modelo, dataset, texto_inicial)
        print(new_refran)
    except KeyError as e:
        print(f"Error al gen texto: {e}")
        print("Asegúrate de que todos los caracteres en texto_inicial estén en el vocabulario del dataset.")

# Visualización detallada de la arquitectura del modelo y valores de las neuronas
def visualizar_modelo_detallado(modelo):
    plt.figure(figsize=(20, 12))
    
    def dibujar_capa(ax, nombre, forma, valores, pos):
        x, y = pos
        ax.text(x, y + 0.1, nombre, ha='center', va='bottom', fontsize=10, fontweight='bold')
        if len(forma) == 1:
            neuronas = forma[0]
            for i in range(neuronas):
                color = plt.cm.viridis(valores[i] / valores.max())
                circle = plt.Circle((x + i/(neuronas-1), y), 0.02, color=color)
                ax.add_artist(circle)
                if i % (neuronas // 5) == 0:
                    ax.text(x + i/(neuronas-1), y - 0.03, f'{valores[i]:.2f}', ha='center', va='top', fontsize=8)
        elif len(forma) == 2:
            filas, columnas = forma
            for i in range(filas):
                for j in range(columnas):
                    color = plt.cm.viridis(valores[i, j] / valores.max())
                    rect = plt.Rectangle((x + j/columnas - 0.05, y + i/filas - 0.05), 0.1/columnas, 0.1/filas, color=color)
                    ax.add_artist(rect)
            ax.text(x, y - 0.07, f'Media: {valores.mean():.2f}', ha='center', va='top', fontsize=8)
            ax.text(x, y - 0.1, f'Std: {valores.std():.2f}', ha='center', va='top', fontsize=8)
    
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Embedding
    embedding_weight = modelo.embedding.weight.detach().cpu().numpy()
    dibujar_capa(ax, 'Embedding', embedding_weight.shape, embedding_weight, (0.1, 0.8))
    
    # Capas del Encoder
    for i, layer in enumerate(modelo.transformer_encoder.layers):
        # Self-Attention
        attn_weight = layer.self_attn.in_proj_weight.detach().cpu().numpy()
        dibujar_capa(ax, f'Self-Attention\nLayer {i+1}', attn_weight.shape, attn_weight, (0.3 + i*0.2, 0.6))
        
        # Feed Forward
        ff_weight = layer.linear1.weight.detach().cpu().numpy()
        dibujar_capa(ax, f'Feed Forward\nLayer {i+1}', ff_weight.shape, ff_weight, (0.3 + i*0.2, 0.4))
    
    # Linear Output
    output_weight = modelo.output.weight.detach().cpu().numpy()
    dibujar_capa(ax, 'Linear Output', output_weight.shape, output_weight, (0.9, 0.2))
    
    plt.title("Arquitectura Detallada del Modelo Transformer con Valores de Neuronas", fontsize=16)
    plt.tight_layout()
    plt.show()

visualizar_modelo_detallado(modelo)

# Análisis adicional de las capas del modelo
print("\nAnálisis detallado de las capas del modelo:")
for name, param in modelo.named_parameters():
    if param.requires_grad:
        valores = param.detach().cpu().numpy()
        print(f"\n{name}:")
        print(f"  Shape: {valores.shape}")
        print(f"  Media: {valores.mean():.4f}")
        print(f"  Desviación estándar: {valores.std():.4f}")
        print(f"  Valor mínimo: {valores.min():.4f}")
        print(f"  Valor máximo: {valores.max():.4f}")
        if len(valores.shape) == 2:
            plt.figure(figsize=(10, 6))
            plt.imshow(valores, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f"Mapa de calor de {name}")
            plt.xlabel("Dimensión 1")
            plt.ylabel("Dimensión 2")
            plt.show()
