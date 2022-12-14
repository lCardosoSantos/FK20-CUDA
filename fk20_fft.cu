#include <stdio.h>

#include "fr.cuh"
#include "g1.cuh"
#include "fk20.cuh"

// Powers 0 to 255 of 5**((r-1)//256)%r

__constant__ fr_t roots[256] = {
    { 0x0000000000000001, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    { 0x33811ea1fe0c65f4, 0x15c1ad4c687f28a2, 0xecfbede342dee7f4, 0x1bb466679a5d88b1 },
    { 0x2cabadec2fe322b8, 0x752c84f315412560, 0x32a732ae1a3b0aef, 0x2e95da59a33dcbf2 },
    { 0x16654b4e3193f385, 0x7dd34dc48b542a79, 0x881dab49e15d3727, 0x6a61dbd781fc6d1c },
    { 0xb750da4cab28e208, 0x501dff643be95635, 0x8cbe2437f0b4b276, 0x7d0c802a94a946e },
    { 0x8e92a08175236817, 0x2e29e9208281d298, 0xb7701b97622a206b, 0xf798c4fc8876304 },
    { 0x7d01866add958780, 0xdf7b4ee66e4caccb, 0x67e701a945e9b713, 0x18ba7658084c71a4 },
    { 0x9062ff9a0538ec84, 0xa05b275c4ae15be4, 0x435ea8ffe19c8c85, 0x496ecbcd74b6124 },
    { 0xd2fc5e69ac5db47f, 0xa12a70a615d0b8e4, 0x293b1d67bc8de5d9, 0xe4840ac57f86f5e },
    { 0xaddcfc96a506dae2, 0x87736de903ce2b7a, 0xaf1b0f713a146c93, 0x346cbb19b5815e67 },
    { 0x1340a28ffe15d019, 0x2c4adbcf1b2f2765, 0xc98da723e86bdc41, 0x2b92cbbd0503595e },
    { 0x64936d794ecac754, 0x4b4e7bab3c93dde1, 0x97e292677efe6e68, 0x192d8efcbc9ffba3 },
    { 0x6207cd2f20df9d12, 0x0081066e578d78bf, 0x725ce48f79bf9cd6, 0x60c7efec34093e9a },
    { 0xe302dccf4853d300, 0xb07c5b52db2a5c58, 0x1875aba524351eaf, 0x1b4018783b6e925b },
    { 0x8dfcd4912667f55f, 0xe801fb897d05b853, 0x10a20368471f6635, 0x64e32898a7c259a8 },
    { 0xf9ed4f03933eef85, 0x30eb83e42ef3f6bf, 0xa9ca21650de7491c, 0x7d1dbabda2a09d6 },
    { 0x9733f3a6ba60eaa6, 0xbd7fdf9c77487ae7, 0xd84f8612c8b6cc00, 0x476fa2fb6162ffab },
    { 0xd0bd794485b1cd59, 0xde1d4b203d557fd3, 0x6b0fd286dc1624d8, 0x1a8dff0465d7945d },
    { 0xf3f90bd2b05e602e, 0x42f464cb4256a3b3, 0xcfdc1da7bf59004b, 0x36e1ebe124834a8f },
    { 0xf882c6f1282fa4bf, 0xf50c3ae2947f2aaa, 0x4e318d58820a2e95, 0x32569ceaf4033df6 },
    { 0x4000b0b3767c9928, 0x19b0deac619bda2d, 0x8b98c6ffcd682aa2, 0x35c60b00588ed2cf },
    { 0x40e4c962f57bba4b, 0xa3ce9513fd14fa73, 0x5bef62d338fc0848, 0x8097452c5c97866 },
    { 0x8117a83fe0dacaef, 0x5d1a65f594b09926, 0x12712448c25af059, 0x2b761bfda43188b5 },
    { 0xf424341daddcab5d, 0xa86c461ae56f072d, 0x5204f44088385855, 0x4aa10d96bee11ebb },
    { 0x8d28c143168bba2c, 0xdb872ef3dce28e78, 0x4e84b3cd7dc75b61, 0x640d097461a2eeaf },
    { 0xfa6ceef222eb879a, 0x2ef304939c366da7, 0x99f61f4617deac54, 0x6d49af90ed12f44 },
    { 0x998731ee0c78437a, 0x38275f891994324c, 0x51ea52ef08f1e06c, 0x4ba357052117e1e1 },
    { 0x1602cb7786c1f5e3, 0x682032b420d1fe8b, 0x3bc2e754e78bbdb0, 0x67eb2fd57acdc0bc },
    { 0xe2c1b6af78029270, 0x28db89e0a4a5765f, 0x7541df17373a0657, 0x4f531b5dc39ecc50 },
    { 0x7372e26d5720949f, 0xaa9ad32419396162, 0x5e3ab4ed8e1c3483, 0x4a42ca6312ad70d },
    { 0xf007bdc150d024b1, 0xbf5354ab6e8c2ab4, 0xe0eb9d91cdecccfb, 0x1f2caa7aeef7d8de },
    { 0x5d8e6990bfa23004, 0xcb1bb332b4e8ce84, 0xd4fbf85ba3c7b295, 0x3c5dbc8a3a58c72a },
    { 0x95df4b360411fe73, 0x1ef4e672ebc1e1bb, 0x6e92a9c460afca4a, 0x4f2c596e753e4fcc },
    { 0x122fe2e0669a2cc8, 0x3050700bdc4b9b68, 0x7a36b26c3985d063, 0x35bc18d05559682 },
    { 0xf6f43227eaade450, 0x614a95d2c3e71de2, 0x388a3903558023a2, 0x2d432e127d4b36ff },
    { 0xcf4a24368337be6a, 0xbbdd88120258390f, 0x4cc9ff4bf19abce5, 0x30e1bc58cb5eb5c4 },
    { 0x3ff630d3f89c4c2c, 0x08ccffd43f1931b6, 0xfaa1f3804fdb73d0, 0x1f3834a0f387add7 },
    { 0x4219652d417991d9, 0x40e18e474ae24746, 0x43e305da5e1bbe7b, 0x1cac59e2fa4a0e2d },
    { 0xf003cd6e73efb611, 0xaabff88c0b7e4246, 0x0cc516e3bb219c82, 0x146d0fdde70e992d },
    { 0x1be69f3270830aad, 0x6862075a6764c01e, 0x447bddbac2fd51ed, 0x5807969225ba3389 },
    { 0x7b9610b1db94977e, 0x6ee1991fee0e510a, 0xc1649310899759ba, 0x205bf9e8afe7612d },
    { 0xb38d064ea8444127, 0x510352b62f3f203e, 0x023718eed874887b, 0x5aeefa03ad3c8255 },
    { 0x2e9a4db5f138dab3, 0x7a1cb8c070f0ce32, 0xbcca660ee327eb37, 0x6a945980dade66ef },
    { 0xd5191d3dde403c3e, 0x71f811d4c6505005, 0xe84835306f7d058a, 0x3ed091c4b9d1142 },
    { 0xb08956f964517062, 0x29c85f7efbd6188b, 0x09768a7087f0cc26, 0x13562b7d27b4c23e },
    { 0xbd7831edbb030fe8, 0x3305eef4a96aad48, 0x6d7b9d657ca55704, 0x3ec15814750a6cbe },
    { 0x2874e1098b7e9b1a, 0xc684d7b3f8b7cf13, 0x7a9970f370ae50cb, 0x15855092e98ac1b4 },
    { 0xec3894fdbbc5e597, 0x0a738f0e49608f7f, 0xe522dc0df640392d, 0x2e916831b2c665dc },
    { 0xae34a81e1cfca2ec, 0x6c7876bf0928e68f, 0xae25b21df6c52b2a, 0x5e73ed8c432405d0 },
    { 0x623d46a32577733e, 0x4c3bea52fcee5c63, 0xd3aadb6799033829, 0x24e340908d912fb4 },
    { 0x3b2bcdccf1ec7b16, 0x6f5f56946340fdfc, 0xbae4d7a5261114f2, 0x55e29e7e1c0691a3 },
    { 0xba7d4348b3e9e6a5, 0x4fa1ce0189080895, 0xafb713389188cd09, 0x2e26afb08780cead },
    { 0xd10899b24dbbf12d, 0xf69f9c8b080ee81b, 0x313b8642525d5075, 0x22dfb19b5e5f8ee9 },
    { 0xfea7c33496a345f2, 0x920d246202280193, 0x0156dd6f2d56b9c1, 0x59a2bad21bb26cbe },
    { 0x8d66b6b5017225bb, 0xcbba2b752da27a4e, 0xd247f1687a9eb754, 0x11c17601029a4196 },
    { 0x5f2fb9c8ee9f3784, 0x5d5e4ac091a8413b, 0xf1b534e07ce71098, 0x55c4b0d68f671db5 },
    { 0xf44e14739932aa33, 0x59e74ba2b75ca477, 0x59d7d47f20aa9f02, 0x37d3508a14adf959 },
    { 0x140700f6f5ad1992, 0x9d27a423c7cc92cc, 0x84be2f7125588a79, 0x3da1160d7fc7387b },
    { 0x523db144c948a873, 0xdf83c14c0425182e, 0x2c14671f92089bc7, 0x19e77e100025cede },
    { 0xf825d2b60a32f87a, 0xb99d89aade182224, 0xe027a83558a1fa00, 0x1df44aec85b8f956 },
    { 0x91e8a1ba3b6f3441, 0xbe66f8a0dc0e6bbe, 0xe1b2c3aa0eb455ff, 0x1a112d20409a835f },
    { 0x64d11e5e40eed6d0, 0x7d51124d9f38a3d2, 0x248cee69c0ae043a, 0x4fb2cafd455e05aa },
    { 0x7675b5beeb6153ca, 0x36c5916205245125, 0xea18ac2f59bed4eb, 0x44057850ce28f03e },
    { 0x625384be62a7a2e8, 0xf63fb9a2d6fcdc22, 0x2c5ded5ceeccb2ad, 0x109de66c7ea5ba6f },
    { 0x8dd702cb688bc087, 0xa032824078eaa4fe, 0xa733b23a98ca5b22, 0x3f96405d25a31660 },
    { 0x97ef4c4862dac48b, 0x2b62a8e248a58739, 0x756db007d73ddfd8, 0x835edd0c55c21a },
    { 0x6358cf18763f4063, 0xd98a9a8b48559c15, 0x3e293867324c9a71, 0x73432a235b185c76 },
    { 0x8902bffd4b07db39, 0x9fc5e9ec9c4b3efb, 0xeeb4135df31350fa, 0x42f60cdeaa382f0 },
    { 0xa4f1f87d90884c5a, 0x7f3accee4e32feba, 0xa84b028e307c9363, 0x22545f6ba59c3994 },
    { 0x0ade8d476f637dd2, 0xf024d57952c2179c, 0xf78e806eb0f4022f, 0x71b597e408eec004 },
    { 0x194e60f3a1520636, 0x7be32750901a3f7f, 0xa495eee2bbe6b01a, 0x57eb8aeb7895fc66 },
    { 0xb0a40e557f4b46ae, 0xe49025299f9d89f8, 0x07e1837d2ec951dc, 0xc0eb7cf8d35960d },
    { 0x223c4fc056e43d9a, 0x9bbabf1e9e172de6, 0xabac70f83e4acd51, 0x10952f0122e8072d },
    { 0xac9567be9a36d522, 0x59d7a64cccac9b12, 0xefacaa48c2c48274, 0x6b54c447ce21b78a },
    { 0x75352d72a0c9539a, 0xa07fcad29e981643, 0xfcf3ec5eda0ddb53, 0x42a2e61a3a2a9b39 },
    { 0x1e268819ba508ef6, 0xd6b8cb20a77d7230, 0x69c5df0b617bd142, 0x2170479ce4654983 },
    { 0x2a0a83854a64dccf, 0x0dc290a30beab8a8, 0x96c0943166661472, 0x11fb33e0bb0973b0 },
    { 0x799bd1773c3aadf2, 0x77bf1d8539e5c50b, 0x2c0cc5b558b3509c, 0x455ed6eec90f1e19 },
    { 0xe9dd22a497765719, 0xc2b901d9b44bf6ce, 0x8e1dd927f6f6bf4a, 0x3dbe994d9fc0040d },
    { 0x71962ed94d9ec469, 0xc21d1aec1b762d2b, 0x6db2b2d9ca2e4902, 0x5ea08340e97d268e },
    { 0x3ec50c9031a36ca3, 0x26bc7f62d13a6e1c, 0x06050b5832d10997, 0x65f6c5837cb5fca2 },
    { 0x2d0f0a19166b00c7, 0xbd8bcd5f497503b1, 0xc54b5ed9c1a33946, 0x41850e645548e4bc },
    { 0x4b94920969459132, 0x3768befea0679e49, 0x1d51a2e03b9c9cf2, 0x1e03d44cddf6abc2 },
    { 0xe7c952e4bd7af7d4, 0x15b7da58c5251abc, 0x278a8c2e3d1fd8e1, 0x5e8542944aa32bee },
    { 0xb14b44a23cfe6f92, 0x4f75c4fd31a9bdfb, 0xd9bbb5adebfdc299, 0x2a8fa6a84eec93c },
    { 0xc6570d2ea95bfd84, 0xdd4c749d2221afab, 0x99c8a193d439a310, 0x1a41230b34c24e22 },
    { 0x24438957b3e25619, 0xb20e24ef2f570c8a, 0xf82295c732d111a2, 0x1a660e58dcdd8ce7 },
    { 0xaadbf867e295fe4e, 0xd72736107d34be5d, 0x13c6ed1276f03221, 0x6388e2b1413ae56b },
    { 0xd64b34dde86304f3, 0xe94910119f300f47, 0xb3c03b15fa3a7826, 0x70822dc00c9fa4b3 },
    { 0x3cd3d2e1c03a844c, 0xf1a7edc3eeb1ea0f, 0x2f9ded387ed31cf4, 0x1257c7495e0793b3 },
    { 0x3c1d97dd9423b5d1, 0x490a9a547af7f720, 0x6dfda5ce9bf2fc0c, 0x67dc39c63dcea51c },
    { 0x8bc1932b0d274a1d, 0x885b416bad06a752, 0x70f081507724a71a, 0x5b78f02be95b1a9e },
    { 0x44568101d845f71f, 0x9d4b0f40448ca04c, 0x6e8ba6f2374a655d, 0x5318150315161c3e },
    { 0x93277736a35fa518, 0xa61d9f02709eb900, 0xa14d12d4e142af50, 0x588bb8a187109e2d },
    { 0x4c191129505eef66, 0xf27edb7e69c459f4, 0xb3e1674c00ddcfee, 0x1638d87eefb9b5be },
    { 0x963a9c566e8bd6b5, 0x5f468edeb6787ad5, 0x26747ffd80262ac3, 0x3dfd92e903126986 },
    { 0x5e544cdf4c887948, 0xfc010d53ccb22542, 0x687a73c92f188612, 0x3b25b475ab91194b },
    { 0xbe92da98becd1980, 0xaac7ab07e301263f, 0xd5f34d908dff438c, 0x104bfa81cc9e333f },
    { 0x2efcfdaf4e265ab7, 0xcaafcb31434545d8, 0x9a54b1f2bebcea8b, 0x2fb2b48988b9dbae },
    { 0x1ea4d56a5d6a3654, 0x824a1bd4592fe155, 0xc62477cbd8d1dc72, 0x3679279f667a6ff2 },
    { 0xf6a1cd7025d071a5, 0xa9bdb5218d05af8a, 0x2ed1234cde5d6927, 0x4ae921e17d3cd87a },
    { 0x2318f1f6844036ac, 0xfe4de56daf67ef10, 0x604297e1df5b7fe9, 0x709701b7108f6c67 },
    { 0x3e0664b6a44982e0, 0x64560ac1281b27b4, 0xd2d28fd9dc0d920b, 0x18fdcd6eb725f684 },
    { 0x064a5280cb9b0a5c, 0x028a0bce9fdfac61, 0x2dca882e26bcc0c1, 0x22de564df61c66ea },
    { 0x9ddf46bac40ac8e4, 0x0bf20a6f5e170989, 0x916f129332ba2dfc, 0x28eb300e9079af0b },
    { 0x87b9261d38ab4a25, 0xc85aad4b9c1dd14b, 0x2b3248a8f8582464, 0x41715c5bb54b12a8 },
    { 0x045697cc52056d9b, 0xc866372d5f7944f1, 0xd986c0c20a39fc66, 0x25b9984a697ff768 },
    { 0x5d970e0b0666e0d8, 0x0ad22b0908293b43, 0xb44d3b688f849246, 0x5b030683301968fd },
    { 0x5a9a50847f96ec43, 0x11f69b4a47af7565, 0xa6231d154bcbeebf, 0x463690b59abcf615 },
    { 0xa87aba4a19b4551b, 0x77ecd8830dcc0a87, 0x29114bed4238dad3, 0x3fa220593e7aa7d6 },
    { 0xc7ea608db8cd652c, 0x404ef4c6bf3fe0c6, 0xf5389b96856be8b1, 0x1df57ca6fc96f67f },
    { 0xf8d0009cc2a6c535, 0xd54537fd0d912eea, 0x5d8cb9d35a162fdb, 0x28de61eb978ffd8a },
    { 0x196986abd51edcb5, 0x708c291cbae2a361, 0xac3f6e2f24bd52fe, 0x6f8c83d4d0a0ae73 },
    { 0x97c7058ee0f32403, 0x60278442906cdcb9, 0x8c826031662df9e7, 0x6c7a15d0bc7ed9cb },
    { 0x8ab3113c021da780, 0x6212655fec382b48, 0x180ea84a5115cfd2, 0x53aa7d875f42d925 },
    { 0xc3adf3d3e5d685bf, 0x13fd86cce858f8c8, 0x8df483f139d30e3c, 0x1c1c8c1e703f7f80 },
    { 0xa0927a327c751043, 0xb830a34e5e4c7a31, 0x413ff88848100458, 0x30d2983de0110e23 },
    { 0xca2e518d542f5ab4, 0x37f819f7fbd18722, 0x5c7bd78eafd2363f, 0x5c5a35118c1d063b },
    { 0x2d787c980313ea47, 0x95f315a6b2d095ff, 0x758318df7c28ca7b, 0x16d4372554761399 },
    { 0x831af13e1a684e43, 0x9a8d4f7d4f608b27, 0x1aaa8bc17630875b, 0x2669eabd406c3a9d },
    { 0x870b11e11ba54d0f, 0x81a2bee506858fc6, 0xbcf95c6ee2bfcfce, 0x596a3d0ddae097e },
    { 0xb6526124c54da696, 0xfcac3a69d3f05c06, 0x34771621130c70cb, 0x7baba3f3484126d },
    { 0x62b851e3d2a34df5, 0x64505ce328284205, 0x604ccbce31863317, 0x663d0aa154b7afb7 },
    { 0x30d3acddb0356ceb, 0x78f1f245ca8edb31, 0xc3916432097245e5, 0x3f57c3bb68888454 },
    { 0xbbf254b932e053a6, 0x72291d47986d31d3, 0xbde19c2cf633387e, 0x5950a37bb55ee341 },
    { 0x2a5e7b785841a1da, 0xeaa07e9cbe5a10af, 0x9589c90d5aa47634, 0x668db67ad287bb0f },
    { 0x0e5070b9bada029b, 0xf244f49758f2e4f9, 0xb5c8af93f751a2bc, 0x27b9f6421e7859bb },
    { 0x7acc968a7ee39687, 0x2b501e5b050695d6, 0x8d1373f244cb0933, 0x38b7e9acd483910f },
    { 0x0001000000000000, 0xec03000276030000, 0x8d51ccce760304d0, 0x0000000000000000 },
    { 0xe0723c72eb58f584, 0xf9140b7174540d1e, 0xf01e6f45ceee8419, 0x21530d469c569786 },
    { 0xeb7d8810073f5d6a, 0x7a807acfac7a7564, 0x1769583e0d6d2c1a, 0x415e4339fb741fc3 },
    { 0xa869ecf2cc0d850a, 0x6a1592642caef3e5, 0x6e952e4835770833, 0x4164d792bef1368d },
    { 0x4222925245f4e2fb, 0x20b2b183f7ec0b19, 0x09b97ddd04dcc909, 0x18780c47d9f8e633 },
    { 0x0deb33b55635d786, 0x64dd6fb6b626c098, 0xb9e31698cf2929dc, 0x22012b7ad6b8e554 },
    { 0x6cfaf4c817c13fe6, 0x30de4d0617413924, 0x57f5f1a3b50f2b7f, 0xc50e1ec7a28bb8 },
    { 0xa099ecb7d80ab778, 0xe565da083310e06d, 0xf6430aea6aa55bf0, 0x7dee6259f2dfb34 },
    { 0x30f9a5d7f0c3cdb2, 0x6faaf80b0c55bd84, 0x3c8c71b5ba9e612b, 0x1ef34257e466b84b },
    { 0xe252f6a3681e0665, 0xfae92a5ff656e571, 0x8213bbd737bf0e62, 0x67b748279a4d3292 },
    { 0xd010a9edb4b638a4, 0x9cc72871c1151445, 0xc4f0a81758d181ac, 0x713a9164dbe6739d },
    { 0xbe012e3d0f88b8f7, 0x6d5dbfac1f1fe4b9, 0xf738c5833b75cdaa, 0x1b02b7b62c9021ab },
    { 0xdbcc4af4074489e8, 0xe6409954448fd0a2, 0x07238a67fd3278fd, 0x1bcee4668304bec2 },
    { 0x49182c727dbb9b2e, 0x3031b0356f6a9b5c, 0xb0c1aa8a59aec67c, 0x51b3482e7e3453a9 },
    { 0x5dd34f3efbca7e49, 0x933c92294a7df311, 0x44915a4a7e636571, 0x32d547e9d6455993 },
    { 0x1848f2442fea2cc8, 0x493dc04931711787, 0xe204d5dce8844248, 0x2cb06e34562bbc52 },
    { 0xdf2b12e5b0330148, 0x51d460a1dbfcd267, 0x8a490835c612d679, 0x20e9cd3a7fca77e3 },
    { 0xc460bf22c6dd910f, 0x2f72d3eac34989c0, 0x6b57d03403d311f6, 0x2789ebc8ca0be3db },
    { 0x198a034b1e705c0e, 0x3c771d65b2ee56d3, 0xe04369454396c389, 0x255c117113c5d5a3 },
    { 0x0d41b8bf60aef96b, 0xd7819e5ef22725f4, 0xb6a386acee69958a, 0x24eb217edb6cee5e },
    { 0x8ad72a76bc61a4ef, 0x94aa3be95461c35c, 0x8506b7660550ec69, 0x595e6c715d7a8ee1 },
    { 0x1985e5010d1e28f9, 0x597feef21aa1455b, 0x4a244131970ee26e, 0x5cf3f7d7a6997c06 },
    { 0xda37c449f8b60f52, 0x3ba63f546586f6d2, 0x2f49c52c6190369d, 0x15e22b8b7d02f5cf },
    { 0x8faa4aea5baef097, 0xfe5e6f6b805b1acb, 0xf1326910bc966667, 0x1616701853308eaf },
    { 0x0727b36db1974ef4, 0x4ee20d3cfc83311c, 0xe7caaba64af67621, 0x28c6d5fd4e2f04c5 },
    { 0xb9f177b76f473cbc, 0xc81a907743c3d4b3, 0xedbfb8fc9a7be5d5, 0x2a441ce244d73adb },
    { 0xa8e747e70c3352b9, 0x1e393a229e5c956e, 0x074231113972d1f3, 0x20b80fa52ca5f2b6 },
    { 0xc40d56452b59c640, 0xe3501d116ae74916, 0x5acd8b4a64b679a2, 0x32e8823696c82086 },
    { 0x7ef262ffa8f8c918, 0x8d28fb6cae51baff, 0x720543223d96f0ac, 0x7aa611dcedcb15b },
    { 0x5b36bed952623e8b, 0x15b19c81ab2696c5, 0x06551ddbbdaeb8d0, 0xae89e2178411af2 },
    { 0xcd4b696e7ea423f1, 0x4c72266b93838c16, 0x34c543b8647ce9de, 0xc721336b5ca226c },
    { 0xba7f245503b95a8d, 0xd34c06153eb88810, 0xbf948a2120473a20, 0x63c21b18f176a5e0 },
    { 0xac159e2688bd4333, 0x3bfef0f00df2ec88, 0x561dcd0fd4d314d9, 0x533bd8c1e977024e },
    { 0x626cf66312dc75d4, 0xc0323eed0b7541f7, 0x2cff5e64aca06bfd, 0x521f0cfc53f02ad },
    { 0xd922fa1a3b513f68, 0xa4170673bd1b99a3, 0x054ba2513a8f6466, 0x36294ce5aa3941ee },
    { 0x6d7e35ad7dffbdb8, 0x92aacd273b23eb13, 0x3b841c065526cbf8, 0x219e386c09e2fd3d },
    { 0x3f0ffba2ea149337, 0x15fb82877b46d20d, 0x398677e14a8cb7e7, 0x581583626bca158c },
    { 0x466b081672596f68, 0x46a918bc0373c6c8, 0x3247210a1e7da36c, 0x1f43e2053bec9fec },
    { 0x5b585be1aadcb79d, 0x7545442a4caa31c6, 0xebafe0c16826da69, 0x4fce7c4e35f66b90 },
    { 0x4ec74d66ce5ce51d, 0x55a5a037f3190196, 0xf74564930e6096c0, 0x6852d1a16317e2ee },
    { 0x26be16cb704b990a, 0x308f60f019fd4009, 0xc6e50c813baa9974, 0x2f00a2325ba21faa },
    { 0xeaa83d0bb9842e37, 0xbd69193792854e59, 0xbec7f66e6610a74b, 0x6d3455f8128b96e1 },
    { 0xee3b38a3813633f1, 0x96ce26afb69019f7, 0x4021f50f21b39e1d, 0x6743acd720403488 },
    { 0xd70edde47d675fa1, 0x9ed7699069b48f58, 0xaddfcf3650346bbe, 0x617ac4567150a5be },
    { 0x35825224af0aaa1d, 0xa379114c0bc2a6e0, 0x7c95a954c7a25068, 0x13c81712a762f460 },
    { 0x2c78c12b5913ffa2, 0xaf9d3446b5219ed9, 0x3f01ab9eac85850b, 0x49e602082ceecbc0 },
    { 0x7a79183fcde843c9, 0x7d5d62341f399e52, 0x84061f9f18bf5ab6, 0x697431feceffb218 },
    { 0x9830635b160a4e50, 0x80863f5f009ecad0, 0xb6794f0e98b62bca, 0x14cce86e2ca5471e },
    { 0x82e68bc0bc50a88f, 0x62a6d195014b6410, 0x468b04010fa5c79f, 0x56f35bb8ed54ae00 },
    { 0x25093e32d48647fc, 0x0b0c65a059be118a, 0x24e66bbc118025ac, 0x456089a064a13d05 },
    { 0x1c2a241e2d08c1c0, 0xef777b4d984f404a, 0x751b65160340510a, 0x522f0091746f546 },
    { 0xcdcd38d6f117f52f, 0x37471747bfef212b, 0xb6548e4a26c46cf6, 0x260e873a1f951adf },
    { 0xc15168ac46d068db, 0xa45d206f9aa2eecc, 0x6f0af7c7244a3da1, 0x5e3f8d3bec0cc346 },
    { 0xfd1d06b2287e2963, 0x1d10182ef83597e2, 0x2bea3d671d909744, 0x5c2bbaf250f91eab },
    { 0xef678c866ccd32d7, 0x897d166d26d7d13f, 0x1810c6122bee51ba, 0x725d1642c7524b47 },
    { 0x32d4d0b2be200c36, 0x4b089f3a743c5027, 0xbc12df9658c7b4a7, 0xa5e89a7a2513fdd },
    { 0x5ba6c1e4335d3495, 0x0cf9b80ec36bb2b5, 0x4f4bc1c2f8689736, 0x20261c76ad9e668d },
    { 0x83a202b9605bdc7f, 0x0a2b4bfeb55dfba6, 0x3d24b851b6e0e294, 0x5b37cd834da0dfe1 },
    { 0x7d6dcabcb43adfeb, 0x98f14a39fc751f56, 0x98af3a8f44a30552, 0x26b8a40efa6dcac0 },
    { 0xd7a9ede4acccf961, 0xa826621eaacc2ecd, 0x805c3d7296833b35, 0x1689d336fe1aa081 },
    { 0xcdd815ee4d942bee, 0xe18a65bc0f8e4d1e, 0x0d10f88231dde2f6, 0x3336a05ea5c70ef5 },
    { 0x5f15555ba7c59481, 0x427b55024e5ba8cf, 0x2edb5dd3926d465f, 0x408b15b1a1011296 },
    { 0xc0ef0c55544da820, 0xa2a2163218d48aa1, 0x4c72f50d0dde0ecf, 0x3cd5239cd9c777f7 },
    { 0x5e501a6aa6138595, 0xea1102eb66f6fce5, 0x36060cc57d5ad557, 0x6d3d8e2034154565 },
    { 0x9fc8017d5166afd3, 0xc54913cc6b4e0c26, 0x787d7d083f1b189f, 0x60b9f524ccbc6d03 },
    { 0xb579a40cf2801b02, 0xd8ebfd640c6d99e1, 0x74ce062206a898e6, 0x698d2aab2ab68646 },
    { 0x520c8c4d748fb509, 0x8b7570b00fca5c9a, 0x4b35861cb1da153a, 0x3d0655665267f097 },
    { 0xe0fda947724a81e9, 0xebbc8a5b856414d3, 0x0b9943d2d6d61529, 0x2ca0b0c5760e32cd },
    { 0x12605a3ecd13f5de, 0xd540042d8d8ef0be, 0x8ee8aa1b9e97d4cd, 0x14be0024e28a13e1 },
    { 0x7b51fa789fe811c1, 0xc02d67a18dc83827, 0xdf3b80de686838f4, 0x420a6d59e26f95d8 },
    { 0x101710f74afb7107, 0x650dca122075b980, 0xf1d522fdb2753b9f, 0x6916332fb92a0335 },
    { 0xbad332c47b502202, 0x9ede4e04f17d1bbe, 0xb3aff7b76fe8a396, 0x2429e6965a8b5056 },
    { 0xc0f1357a508e5e7b, 0x911f590ef73ba2bd, 0xaf59d0fc7261da72, 0x58c400aba73798bf },
    { 0x34d926b66d9a40d9, 0xd88feed2f822e3f6, 0x0b4704bcfac11d52, 0x12ea7051d2cfcef6 },
    { 0xe9b77e2a455758d3, 0x25acb6a751c4cf62, 0x54f132dd768cbcd0, 0x2fbb702adadc3bea },
    { 0x45a8d91e955983d1, 0xd4f39aabdcbb7d07, 0xe05891c9ff9c3987, 0x75ded933c287503 },
    { 0x9cc2c16c38357332, 0xd74c9bd5d6faba52, 0xf21979ec507b8b9a, 0x374e4d8c8ac99dae },
    { 0x2393abcd485fee6d, 0x4ed19bc8cbd2faac, 0x91f84ff7a4d48332, 0x493e63042b0b017f },
    { 0x0f15758baad3fcf4, 0xb06cbfc87f549eb2, 0x782fd931be0cba8a, 0x18e17eef82a73dc4 },
    { 0x7d8a7a6c8da4a98a, 0x121959d9d4d000d5, 0x43c94f59a6eb37f3, 0x278e695f56d8f6c5 },
    { 0xc9f39f658c9620b3, 0x944f1b07b3c56074, 0x7e7d03f9e6ac83bc, 0x230d17191423f48d },
    { 0x05e2e6cb0db28404, 0x557f4e7f70ec86ac, 0xe07032ac4dd9f486, 0x222d71b2bcbcebb0 },
    { 0xbf7579b9d2eebd4d, 0x4d693fdc8713f747, 0x963edac49c2696da, 0x178b92aeafc59bc1 },
    { 0xd098777bae4c5131, 0x401d8a62b9c4f076, 0x87b5b1048c572910, 0x730e62c281967ad8 },
    { 0x18d303a266992f63, 0xf495392351a789a3, 0x25302664820cb7fb, 0x58e6bb22f520df2 },
    { 0x8aab34fd0a699f92, 0x0b5b986955de3d63, 0xbb3dd6f817e65025, 0x730f5244f60d187b },
    { 0x16a0c798f162eacb, 0x788db8d2298d3f8e, 0xc48ba1f92b1fec5c, 0x6bbacf811bf1365a },
    { 0x46add4de51661716, 0x320e8aaa2c0adab6, 0xd332c8edcb31b9c5, 0x6f8c17a81ec992db },
    { 0x73ea3fa3e6fe310f, 0x4eb73f5d99585805, 0x412d801131c542d2, 0x68f0ba2461933c32 },
    { 0xb5cc4b34b05aff2c, 0xdbeecf7cd0cd4949, 0x668000c77dfd2ab5, 0x2d17c104bb6ae704 },
    { 0x08993b54226b7f93, 0x30dcd43dec9acd71, 0xfa481d537345cf65, 0x42083f58561023a7 },
    { 0xe304a47cfe2c39f0, 0xe2b0a7d87187c5c0, 0x293082360528eebe, 0x3b4acba79e856c22 },
    { 0x9b173ba8dd620212, 0xe1ffe63827d53336, 0x2b52573b968efab1, 0xc3fd4bf2035cba6 },
    { 0xb16c92903c30681c, 0x44df8723cf6834a9, 0x4a7055b0fa4b32af, 0x2a337af602a41b0e },
    { 0x8bd63e0485773dba, 0x83be0261fa1ef074, 0xed257db1ed5ff5da, 0x6cebc5875ac931fb },
    { 0x94e53fb564bdd489, 0xb0f3bdbc0f422b0a, 0x943f2b9abaa7250d, 0x5c6170ead42b9162 },
    { 0x807b811a823f728d, 0x967a6b6cfb0faca4, 0x5ccd4631f16edba4, 0x1edc919ec91f38ac },
    { 0xf7a3f674f9b19bf4, 0x631c5b62960ddd7b, 0x400753b97514ed76, 0x1a308a0fee2cab3a },
    { 0xfdde6c4bcc3c47ac, 0xe52fb0fde29479de, 0x051f83121c342bf6, 0x3d6a7ee321cfb3ba },
    { 0x0c0871ffc19b100f, 0x2aa959dcb76fe539, 0x894506e8611524fe, 0x2f73ce8ba2c48aa4 },
    { 0x241de8ab6c75208d, 0x87c0551ac0e33765, 0x5eec2340271598ef, 0x59449d9a30240186 },
    { 0xe0a8ac1b333b6dd3, 0x138cca56e78e96b9, 0xe8378be639ffa7e0, 0x323f735d04dc0040 },
    { 0x85caa34fd9b2b6ce, 0xe540e7eb01aad13b, 0x337b8c94b4e3a208, 0x7275b1d9346b1f74 },
    { 0x12e39b8b4c1ce30a, 0xacd388dec8bdb5c9, 0x2f946b3f886734d3, 0x512a38f4465cd083 },
    { 0x7ebf2fcc0f5611ae, 0x262724b71d050aa9, 0x1e97f596a4c8da48, 0x2e3e440d3d981efb },
    { 0x2b436d7a469e3045, 0x7a6d63f8b2850344, 0x0c095cf1e3315298, 0x4c75d3a13151f5da },
    { 0xcca54ea0331495d4, 0x05b1ae62415a9910, 0xb40ec2b38999673e, 0x163daecf5af2f4ab },
    { 0x2f3ec47bd8b0ba51, 0xf4de1994dace1ccf, 0x21c1bcc99837c8b4, 0x480fec2473e12abc },
    { 0xe9dc91a650a90e49, 0x5cae4a21d39a23be, 0xf2f3e98766f8c918, 0x16c359d76ca02af7 },
    { 0xdf259f8f0b5c45d6, 0x29749ce3d6806cc2, 0x67409b01de51fdb5, 0x12e9de85005026e },
    { 0xad5b3bb2b65b60e3, 0x1140515e8a158ded, 0x18fae2b2b0ec6a98, 0x5ff8b7ffb85398c },
    { 0x187a081bc020c9e9, 0x88b1cebef9a7dbaf, 0x2eb22d78a01a1491, 0x48c649dce1168ed7 },
    { 0x11a43132b1888c36, 0x0e6466930923ab78, 0x8f907adac98af548, 0x6f70f5e67a06fbd8 },
    { 0x92b4c291774613b8, 0xe87ab3a7d22d6c55, 0xa9326e0004f6e7c7, 0x196f79855eaa576c },
    { 0x3e16064aa9a77770, 0xe401cd8bef1b2260, 0x7fb60b67a741a4ee, 0x6635214dfb1c37d9 },
    { 0xbd7360167f49690f, 0x55ed617941b103b9, 0x1486518d8c5f9689, 0x40b5b7898c657ed6 },
    { 0xaecf3d3d99feda43, 0xbaecd9567808eac2, 0x7b734cba89c47734, 0xb55963f1644f605 },
    { 0x2ee4c0317b41459b, 0xbab9bbd9b0a3590b, 0xb5c82a603c67a3a3, 0x16fd8c1a3548e372 },
    { 0x85fcf33c21a24ad6, 0x9d80256afc11e71d, 0x4dc142b14aa24926, 0x56dacb61c12c11e7 },
    { 0x70217c6264ef06a7, 0xf3b3f165d5541509, 0xa1d384187ac0cbd3, 0x68a5b2d1c312e47b },
    { 0x476e05bf67d4973c, 0xaf9f1f01e2bbf0ac, 0xdcd9ce528178852e, 0x5a50cc64ae610371 },
    { 0xf65070e6d99ba93c, 0x73680114a074f1ba, 0xdc895e94a770ff33, 0x2096fb30ee127ca0 },
    { 0xb6ca42cf7e8096b6, 0x49b53e5a66157cff, 0xc80356e65d94e358, 0x245266ba46b38c19 },
    { 0x3df47cf236da17e9, 0xe92d7dae9287cd04, 0xc579b994cbef0c2c, 0x1306f534d4599079 },
    { 0xa8e45d0230d79fe6, 0x3d4dee80538980ec, 0xe1f7cc23d4a0a0fa, 0x5a56acc5d6a40c8d },
    { 0xbabe46ffd11fbcb3, 0x7d268d78c59e5b00, 0xf93aebe3fc216f65, 0x6a9c4100d476d6f3 },
    { 0x16a9e3e795d08887, 0x9380f5ffdc6ac2e3, 0xa189d7e4af45dab5, 0x688ba2e026f2459 },
    { 0x04caf2400fd5ee93, 0x376c95434baa21c0, 0x3cc485d9d1f4893b, 0x5ae601e4a8a5521a },
};

extern __shared__ g1p_t t[];   // Workspace in shared memory

__global__ void fk20_fft(g1p_t *output, const g1p_t *input, int stride) {
    // No multi-dimensional layout, only one FFT of size 512 for now

    if (gridDim.x  !=   1) return;
    if (gridDim.y  !=   1) return;
    if (gridDim.z  !=   1) return;
    if (blockDim.x != 256) return;
    if (blockDim.y !=   1) return;
    if (blockDim.z !=   1) return;

    unsigned tid = threadIdx.x;  // Thread number
    unsigned l, r, w, src, dst;

    // Copy inputs to workspace

    src = tid;
    // dst = 9 last bits of src reversed
    asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(dst) : "r"(src << (32-9)));

    g1p_cpy(t[dst], input[stride*(src)]);

    // Extend input vector

    dst = 2*tid+1;
    g1p_inf(t[dst]);

    __syncthreads();

    //// Stage 0

    w = 0;
    l = 2 * tid;
    r = l | 1;

    //g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 1

    w = (tid & 1) << 7;
    l = tid + (tid & -2U);
    r = l | 2;

    if (w) g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 2

    w = (tid & 3) << 6;
    l = tid + (tid & -4U);
    r = l | 4;

    g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 3

    w = (tid & 7) << 5;
    l = tid + (tid & -8U);
    r = l | 8;

    g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 4

    w = (tid & 15) << 4;
    l = tid + (tid & -16U);
    r = l | 16;

    g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 5

    w = (tid & 31) << 3;
    l = tid + (tid & -32U);
    r = l | 32;

    g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 6

    w = (tid & 63) << 2;
    l = tid + (tid & -64U);
    r = l | 64;

    g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 7

    w = (tid & 127) << 1;
    l = tid + (tid & -128U);
    r = l | 128;

    g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    //// Stage 8

    w = (tid & 255) << 0;
    l = tid + (tid & -256U);
    r = l | 256;

    g1p_mul(t[r], roots[w]);
    g1p_addsub(t[l], t[r]);

    __syncthreads();

    // Copy results to output

    src = tid;
    dst = src;
    //asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(r) : "r"(l << (32-9)));

    g1p_cpy(output[stride*dst], t[src]);
    //printf("%3d -> %3d\n", l, r);

    src = tid+256;
    dst = src;
    //asm volatile ("\n\tbrev.b32 %0, %1;" : "=r"(r) : "r"(l << (32-9)));

    g1p_cpy(output[stride*dst], t[src]);
    //printf("%3d -> %3d\n", l, r);
}
