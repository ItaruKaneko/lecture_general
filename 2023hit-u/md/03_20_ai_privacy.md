# AIとプライバシー

- AI に対するプライバシーはどう保護すべきだろうか？
- AI によるプライバシーリスクにはどんなものがあるだろうか?

シャーロックリスク

- [医療情報ビッグデータ分析におけるシャーロックリスクの考察とその形式化](pdf/IPSJ-EIP20088015.pdf)
- [大量ウェアラブルデバイスと大規模生体情報時代におけるAI機械学習のシャーロック問題への対策としてのプライバシーエージェントのDockerベクトル化](pdf/IPSJ-EIP22095001.pdf)


AIによるプライバシー侵害がこれまでと異なる点はAIによる情報の精緻化、明確化がおこりうることである。

この現象は、ノイズキャンセリングという技術と似ている。もともと与えられたデータは一見ノイズに埋もれているように見える。しかし情報を含まないノイズだけのデータを別途与えると、AIはノイズをキャンセルして、情報を取り出すことが可能である。

このようなシチュエーションでは、過去に安全だとされ、本人の許諾を得て公表された情報から、機微な情報が抽出される、ということが起こり得る。

医療情報については以前からこのようなことが問題となりえると認識されていたので、ダイナミックコンセント(動的同意)という概念が存在し一部実施されている。動的同意では一旦同意した情報提供を、状況の進展によって取り消しすることが可能である。

AIが情報を利用する場合にも、個人情報保護に動的同意の機能を安全に実装することが必要である、という主張もある。

議論

- シャーロックリスクはどの程度深刻だろうか
- このようなリスクは現在のプライバシー保護のフレームワークで回避可能だろうか？
- ノイズデータを与えることによる情報漏洩についてどのような保護手段があるだろうか
- 社会制度としてどのような仕組みを準備すべきだろうか