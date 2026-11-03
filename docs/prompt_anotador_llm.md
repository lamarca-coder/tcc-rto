# B9: prompt para o chat (um só, vale para as 4 partes)

Como usar, para cada parte (1 a 4):
1. Abra uma conversa nova com o Sonnet 5.5.
2. Anexe só o arquivo da parte (por exemplo, `b9_parte1.xlsx`).
3. Cole o prompt abaixo inteiro, sem mudar nada.
4. Baixe o arquivo de saída e salve em `rotulagem/b9/`.
5. Se a conversa travar no meio, abra outra, anexe o mesmo arquivo e acrescente no fim do prompt: "Comece do id_rotulagem X", com o primeiro id que faltou.

```
TAREFA: anotar os textos da planilha anexa usando o livro de códigos abaixo.

REGRAS DE EXECUÇÃO
1. Leia a planilha anexa (colunas id_rotulagem, plataforma, tipo, texto, contexto). Ignore colunas de rótulo vazias.
2. Antes de ler cada texto, faça html.unescape (por exemplo &gt; vira >).
3. Você deve LER e julgar cada texto individualmente. É proibido usar regras de palavras-chave, listas de termos, modelos de sentimento ou qualquer código para decidir o rótulo. Código só pode ser usado para ler a planilha, montar o arquivo e conferir contagens.
4. Processe em lotes de 50 linhas, em ordem crescente de id_rotulagem, até acabar a planilha. Ao fim de cada lote, confira que existem exatamente 50 ids (ou o que restar no último lote), sem repetição e sem falta, e que todos os valores estão nas listas permitidas. Entre um lote e outro, escreva só uma linha: "Lote N ok (ids X a Y)".
5. Em caso de dúvida, escolha o valor mais próximo do livro, marque confianca = baixa e explique em observacao (máximo 15 palavras).
6. Saída: um arquivo Excel chamado rotulos_llm_ seguido do nome do arquivo anexo (por exemplo, rotulos_llm_b9_parte1.xlsx), com as colunas id_rotulagem, alvo, sentimento, posicionamento, confianca, observacao. Uma linha por texto da planilha, nada além disso.
7. No fim, mostre só: total de linhas, contagem por sentimento, por alvo e por posicionamento, número de linhas com confianca = baixa, e qualquer id que não foi processado.

LIVRO DE CÓDIGOS (v3)
Você é um anotador de textos em português sobre trabalho presencial, remoto e híbrido. Rotule o que o AUTOR expressa, não a sua opinião. Siga a ordem: relevância, alvo, sentimento, posicionamento.

VALORES (escreva exatamente assim):
- alvo: presencial | remoto | hibrido | geral | irrelevante
- sentimento: positivo | negativo | neutro | irrelevante
- posicionamento: pro_RTO | anti_RTO | neutro_misto | irrelevante
- confianca: alta | media | baixa
Se o texto é irrelevante em uma coluna, é irrelevante nas três.

RELEVÂNCIA (estrita): o texto só é relevante quando o autor avalia, defende, critica ou compara um modelo de trabalho (presencial, remoto, híbrido) ou a obrigação de voltar ao escritório. São irrelevantes, mesmo citando home office: perguntas práticas ("como faço para trabalhar de home office?"), logística de vida (moradia, mudança de cidade, ergonomia, móveis, onde fazer reunião), busca de vaga ou salário, anúncios e propaganda (inclusive de móveis para home office), relatos do dia a dia em que o home office é só o cenário ("to de home office vendo TV"), música, risada ou emoji sem conteúdo, texto ininteligível.

ALVO: o modelo de trabalho que o autor avalia. presencial = escritório ou obrigação de comparecer; remoto = home office; hibrido = modelo misto; geral = compara dois modelos ou fala do tema sem foco. Com vários modelos, vale o foco da avaliação.

SENTIMENTO: tom emocional do texto inteiro, independente da posição. positivo = elogio, alívio, entusiasmo; negativo = crítica, raiva, indignação, frustração, tristeza, sarcasmo crítico; neutro = informação, pergunta ou mistura equilibrada. Ironia vale pelo sentido pretendido. Emojis contam. Palavrão não implica negativo. Quando elogio e crítica pesam parecido no mesmo texto, use neutro e anote em observacao; a posição é decidida à parte.

POSICIONAMENTO sobre o retorno ao escritório (RTO):
- pro_RTO: defende ou aprova a volta ou a exigência da empresa, ou faz um juízo geral a favor do presencial ou contra o remoto como argumento para voltar.
- anti_RTO: rejeita a volta ou a obrigação, diz que sairia se fosse obrigado, ou faz um juízo geral a favor do remoto ou contra o presencial ("o escritório virou coisa do passado", "ficar sentado no escritório não mede produtividade"). Inclui indignação com o fim do home office. Crítica à obrigação de ir ao escritório, mesmo no híbrido, é anti_RTO.
- neutro_misto: notícia ou pergunta sem lado, visão equilibrada, defesa do híbrido sem atacar nenhum lado.
- Juízo geral x conveniência pessoal: relato de conveniência pessoal sem generalizar ("em casa o dia rende melhor para mim", "economizo com transporte") é neutro_misto, não anti_RTO. O mesmo vale, invertido, para pro_RTO.
- Preferência pessoal declarada conta como posição, a menos que o texto a relativize e tolere o outro lado: então neutro_misto.
- Preferência só inferida pelo comportamento (por exemplo, aceitar menos salário para continuar em casa), sem o autor afirmar ou criticar algo sobre o escritório, não conta como posição: neutro_misto.
- Relato de experiência pessoal ou pedido de conselho, sem defender nem criticar a volta, é neutro_misto, mesmo com sentimento negativo.

CASOS DIFÍCEIS: indignação em defesa do remoto = negativo + anti_RTO. Notícia ou anúncio da empresa = neutro + neutro_misto. Pergunta = neutro + neutro_misto, salvo lado claro. Texto longo: use a conclusão do autor ou o sentimento predominante; sem isso, neutro_misto. Linhas que começam com ">" são citação de outro usuário: ignore-as e rotule só o que o autor escreveu.

observacao: no máximo 15 palavras, só quando o caso for de borda.
```
