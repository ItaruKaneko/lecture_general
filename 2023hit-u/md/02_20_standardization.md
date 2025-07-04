# 標準化
## ネットワーク外部性

標準は管理された独占である。

標準が作られると仕様の競争は排除されるからだ。

標準化が求められるのは、その分野にネットワーク外部性があるからだ。

メディアにはネットワーク外部性がある。
そこで、相互運用可能なメディアのセグメントが発生すると、もしそれが一定以上普及すれば
加速度的に相乗効果で、周辺のメディアをとりこみながら拡大していく。

あるカテゴリーに複数のメディアが並立すると、きびしい競争がおこり、
どちらか一方が淘汰される、ということがよくおこる。

主な規格競争、収斂をあげる

 - βマックス vs VHS
 - 8トラックカセット vs コンパクトカセット
 - HDDVD vs BlueRay Disc
 - MP3 vs AAC
 - BSD vs Linux
 - IBM 対 他7社
 - IE vs Netscape
 - IE vs Chrome

## 独占 vs 標準化

規格の収斂は、競争によっても、標準化によっても達成できる。
市場競争による規格の収斂は「事実上の標準化」「de-facto standard」と呼ばれることがあるが、標準とはもともと法の力による公的なものなので、「事実上の標準化」はあくまで事実上の標準化であって標準化ではないから、ここでは独占と呼ぶことにする。
(無論適法な範囲)

独占と標準化はどちらもネットワーク外部性の利益を増大する。

独占と公的標準の得失はなにか?

独占は効率の高さと経済的な合理性という点で公的な利益がある。

多くの独占は、的確なタイミングで十分な投資を行うことで実現する。

例:

- windowsの開発 (windows 3.0までは無償で配られていた)
- amazon, google (最初の数年間は赤字)

市場原理に従って投資時期と投資額が最適化されるためだ。

一方独占が実現した後で独占を利用して利益を得るために、価格が割高になるという欠点があるとしばしば言われているが、経済学によればこれは必ずしも正しくない。

需要曲線が価格弾力性があるものであるならば、価格をあげすぎると需要が減る。その際の最適な価格は、供給が複数あっても一つであってもそれほど変わらない。

さらに供給側が複数になる場合設備の重複が生じ、「価格を下げたくてもさげられない」のでかえって価格が高くなることも十分ありえる。たとえば電力会社が2つあったとすると、競争により価格が低下する効果と、発電設備の規模が小さくなることで高くなる効果は相殺してしまうだろう。


一方、公的標準にもメリットとデメリットがある。

公的標準は非営利の標準化機関が作成し、企業は公平に利用できるから、標準化がつくられても競争は残る。

一方、標準化機関自体は独占的な非営利法人であるから、
利潤を追及しないが、そのため非効率でもある。

標準化がどれだけスローであっても、標準化機関には損失はないから、標準化の課程はきわめて遅い。

さらに標準化機関は標準を作ることを名目に存在しているため、標準を多く作る、という方向に動機付けられがちで、そのことが不要な標準を多く作ることにつながる。競争による独占では、統一規格は非常に必要性が高いものにたいしてしかつくられない。

## 標準と知的財産

メディア標準はしばしば知的財産紛争のリスクを持つ。
JPEG標準は、ISOの標準化ルールによって、すべての特許が無償ライセンスされるという宣言とともに、標準化された。標準化当時はこの標準がビジネス的に大きな価値を持つとは考えられていなかった。
やがてJPEG標準が、デジタルカメラの標準フォーマットとなり普及した後で、JPEGの基本特許の一つを保有する企業が、特許料を徴収すると宣言した。

標準化に知的財産権が含まれる場合は、FRAND が求められている。
JPEG, MPEG は FRAND 条項が適用された初期の事例の一つだ。
少し古くなるが、下記資料にその経緯をまとめた。
`

1. [レオナルドキャリリョーネ、 IT・ソフトウェア特許の新潮流 ～活用・防御から標準化まで～：コラム. 特許とMPEGの25年 -特許はどのようにMPEGを助け，また妨げたか-](http://id.nii.ac.jp/1001/00090032/)


1. 標準的技術における特許処理のケーススタディと課題
[->link](http://id.nii.ac.jp/1001/00098394/)
なお、歴史的には特許のない技術を標準に採用する、という方法がとられていた。しかし今日たとえばHEVCに含まれる特許の数は数千以上である。

## 標準化と法

今日こうした事態を防ぐためのさまざまな方法がとられているが、様々な要因が複雑に絡むから、細心の注意が必要である。

独占によるものも、公的標準によるものも、標準規格は競争を阻害するから、場合によっては不公正競争として規制される。

この境界は複雑に入り組んでいる。

特に近年の標準化ではパテントプールの形成が不可避である場合が多い。

そこで公正取引委員会は、規格の統一が不正競争となる場合を定めている。

[標準化に伴うパテントプールの形成等に関する独占禁止法上の考え方](https://www.jftc.go.jp/dk/guideline/unyoukijun/patent.html)


標準化と知的財産の問題は1990ごろかの進展が大きい。そのあたりの事例は以下にまとめた。
[IT・ソフトウェア特許の新潮流 ～活用・防御から標準化まで～：6. IT・ソフトウェアの標準化と特許 -インターネットが変えた標準と特許の関係-利用統](http://id.nii.ac.jp/1001/00090031/)

## UNIX とオープンソース

UNIX がオープンソースムーブメントに果たした役割は大きい。しかしそれは、UNIX がオープンソースだったから、ということではないことはたまに忘れられてしまっているように感じる。

UNIXの著作権、特許の完全な公開が2010年ごろまでされなかったことがオープンソースの強い動機になったことは、記憶されるべきだ。UNIXとその膨大なソフトウエア資産を社会が共有するための障害となった知的財産権への抵抗が、オープンソースムーブメントを生み、BSD, Linux といったUNIXの知的財産を回避したシステムを生んだ。

GNU は Gnu is Not Unix、Linuxは Linux is Not UniX の略であるとも言われる。(GNU は正式名称、Linuxは通称)