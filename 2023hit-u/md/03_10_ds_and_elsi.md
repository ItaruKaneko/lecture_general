# データサイエンスとELSI

## 1. 確率論は比較的新しい

- 微積分、幾何学 ギリシャ時代から存在
- 確率論  16世紀、19世紀(理論化)

データサイエンスを基礎としてどうELSIに取り組むべきだろうか。

この議論のために、まずデータサイエンスはどんな体系であるか、ELSIは何を目指すべきか、ということを確認したい。

データサイエンスは確率論、統計理論の延長線にあり、さらにその応用の一つとして人工知能分野の成果が著しい。確率、統計、データサイエンス、人工知能は、現在もっとも注目されている理論分野の一つだ。

しかしこの分野は比較的新しい分野である。情報理論と同じく、数百年の歴史しかない。

数百年の歴史は十分に長いと思ったなら、他の学問分野と比較してみるとよい。
建築学や様々な工学分野は数千年の歴史を持つ。
数学分野でも、ピタゴラスの定理は、実はピタゴラスが発見したわけではなく、インドで知られていた数学がギリシャに伝わったものである。この定理は数千年前から知られていたと言われている。現在の幾何学は、ユークリッド幾何学と呼ばれているようにギリシャ時代にはすでに確立していた。ライプニッツやニュートンにより近代的な数学分野となった微積分学も、ギリシャ時代にすでに考え方自体はあったのだ。

それにくらべると、今日の確率論の萌芽はどんなに遡っても16世紀ごろ以前にはみられない。
現代の確率論に匹敵する精度の議論は、19世紀にようやく記録に残るようになった。そのような記録も原理もそれより前の時代にはみられない。

なぜ新しい時代にしか確率の議論はされていないのだろう。

## 2. なぜ確率論は新しいのだろうか?

- 非確定的 真実は神にしかわからない
- 実験的に証明できない 得られるのは標本

確率論は最初のうちは科学の理論としては受け入れにくい傾向があった。なぜそうかといえば非確定的で確定的には反証可能ではない理論だったからだろう。

どういうことか？

運動方程式は大砲の着地点や天体の運動として反証可能だ。運動方程式が求める天体の軌道に疑いがあれば、精密にその軌道を測定すればよい。もし運動方程式とことなる軌道を動けば、方程式は誤りだと反証可能だ。
一方コインの表が出る確率が1/2であることを反証することは難しい。たとえコインを10回トスして10回共表であっても、確率が1/2であることの反証にはならない。確率を仮説として受け入れて、実際におこる事象が理論の予測にうまく適合している、ということを確認できるだけだ。
ある事象が0.5の確率でおきることを反証することは不可能だ。

反証可能性についてとりあげると、おそらく物理学を学んだ人は、「量子力学もそうではないか」と思い出すかもしれない。まさに量子力学は確率的な理論であるため、登場時には科学ではないのではないか、という疑問をもたれることになった。

しかしある意味反証可能である、というのは確率論も量子力学も同じで、コイントスであれば試行の数を増やせば増やすほど表がでる割合は限りなく1/2に近づくし、量子力学でも観測される現象が増えれば、非常に高い精密度で現象がおきる頻度は理論が導く値に限りなく近づくわけだ。

また確率論が最初に論じられたのはサイコロの理論であった。これはようするにギャンブルの理論である。
ギャンブルが多くの人の関心時になるためにはそれだけ生活に余力がなければならない。人類が余剰生産力を獲得し、遊興に十分な時間をかけられるようになったのが16世紀以後というわけである。

確率論自体は精密で、ある意味では反証可能だ。が、確率自体は理論上の仮説であり直接反証可能ではない。

現代の科学の基礎である量子力学も確率論や統計学の基礎がなければ、理論化が難しかったように思うが、これらがギャンブルの科学というところから発達したとすると面白いと思う。貨幣経済が発達し、労働の対価ではなく、知恵によって富を得る経済活動が生まれ、特に確率的な現象によって富を得ることがきちんとした仕事として認知されるようになることが、確率論の確立に必要だったのだろうか。