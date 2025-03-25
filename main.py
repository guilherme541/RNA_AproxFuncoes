# pip install numpy matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MaxAbsScaler

# Parâmetros configuráveis
ARQUIVO = 'teste5.npy'  # Arquivo a ser processado
ARQUITETURAS = [
    (10,),          # Uma camada oculta com 10 neurônios
    (15, 5),        # Duas camadas ocultas com 15 e 5 neurônios
    (20, 10, 5)     # Três camadas ocultas com 20, 10 e 5 neurônios
]
ITERACOES = 400     # Número de iterações para treinamento
ATIVACAO = 'tanh'   # Função de ativação
NUM_EXECUCOES = 10  # Número de execuções para cada arquitetura

def executar_simulacao(x, y, arquitetura, iteracoes=400, ativacao='tanh', num_execucoes=10):
    """Executa a simulação várias vezes para obter média e desvio padrão"""
    erros = []
    melhor_erro = float('inf')
    melhor_modelo = None
    melhor_y_est = None
    
    print(f"Executando {num_execucoes} simulações para arquitetura {arquitetura}...")
    
    for i in range(num_execucoes):
        print(f"  Execução {i+1}/{num_execucoes}")
        
        # Configuração da rede neural
        regr = MLPRegressor(
            hidden_layer_sizes=arquitetura,
            max_iter=iteracoes,
            activation=ativacao,
            solver='adam',
            learning_rate='adaptive',
            n_iter_no_change=iteracoes,
            verbose=False
        )
        
        # Treinamento
        regr.fit(x, y)
        
        # Predição
        y_est = regr.predict(x)
        
        # Calcula erro médio quadrático
        erro = np.mean((y - y_est) ** 2)
        erros.append(erro)
        
        # Salva o melhor modelo
        if erro < melhor_erro:
            melhor_erro = erro
            melhor_modelo = regr
            melhor_y_est = y_est
    
    # Calcula estatísticas
    media_erro = np.mean(erros)
    desvio_padrao = np.std(erros)
    
    print(f"  Média do erro: {media_erro:.6f}")
    print(f"  Desvio padrão: {desvio_padrao:.6f}")
    
    return {
        'arquitetura': arquitetura,
        'media_erro': media_erro,
        'desvio_padrao': desvio_padrao,
        'erros': erros,
        'melhor_modelo': melhor_modelo,
        'melhor_y_est': melhor_y_est
    }

def plotar_resultados(x, y, resultado, nome_arquivo):
    arquitetura = resultado['arquitetura']
    modelo = resultado['melhor_modelo']
    y_est = resultado['melhor_y_est']
    
    plt.figure(figsize=[14, 7])
    

    plt.subplot(1, 3, 1)
    plt.title('Função Original')
    plt.plot(x, y, color='green')

    plt.subplot(1, 3, 2)
    plt.title(f'Curva erro ({modelo.best_loss_:.5f})')
    plt.plot(modelo.loss_curve_, color='red')
    

    plt.subplot(1, 3, 3)
    plt.title('Função Original x Função aproximada')
    plt.plot(x, y, linewidth=1, color='green', label='Original')
    plt.plot(x, y_est, linewidth=2, color='blue', label='Aproximada')
    plt.legend()
    
    plt.suptitle(f"Arquivo: {nome_arquivo}, Arquitetura: {arquitetura}\n" +
                f"Média do Erro: {resultado['media_erro']:.6f}, Desvio Padrão: {resultado['desvio_padrao']:.6f}")
    
    plt.tight_layout()
    
    plt.savefig(f"resultado_{nome_arquivo.replace('.npy', '')}_{'_'.join(map(str, arquitetura))}.png")
    plt.show()

def processar_arquivo(nome_arquivo, arquiteturas, iteracoes=400, ativacao='tanh', num_execucoes=10):

    print(f'Carregando arquivo: {nome_arquivo}')
    arquivo = np.load(nome_arquivo)
    x = arquivo[0]
    scale = MaxAbsScaler().fit(arquivo[1])
    y = np.ravel(scale.transform(arquivo[1]))
    
    resultados = []

    for arquitetura in arquiteturas:
        resultado = executar_simulacao(x, y, arquitetura, iteracoes, ativacao, num_execucoes)
        resultados.append(resultado)
    

    melhor_resultado = min(resultados, key=lambda r: r['media_erro'])
    
    print("\n" + "="*70)
    print(f"RESULTADOS PARA {nome_arquivo}")
    print("="*70)
    print(f"{'Arquitetura':<20} | {'Média do Erro':<15} | {'Desvio Padrão':<15}")
    print("-" * 70)
    
    for resultado in resultados:
        arquitetura_str = str(resultado['arquitetura'])
        print(f"{arquitetura_str:<20} | {resultado['media_erro']:<15.6f} | {resultado['desvio_padrao']:<15.6f}")
    
    print("\nMELHOR ARQUITETURA:")
    print(f"  {melhor_resultado['arquitetura']} com erro médio de {melhor_resultado['media_erro']:.6f}")
    
    plotar_resultados(x, y, melhor_resultado, nome_arquivo)
    
    return resultados, melhor_resultado




if __name__ == "__main__":
    resultados, melhor = processar_arquivo(
        ARQUIVO, 
        ARQUITETURAS, 
        ITERACOES, 
        ATIVACAO, 
        NUM_EXECUCOES
    )