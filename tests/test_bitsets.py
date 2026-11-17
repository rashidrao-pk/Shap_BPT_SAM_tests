"""Wide label bitsets, configurable guide penalty and dtype-safe labels in the BPT builder (T1)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/golden')]
import shap_bpt  # noqa: E402
from shap_bpt import bpt  # noqa: E402
from make_bpt_golden import tree  # noqa: E402

GOLDEN = np.load(ROOT / 'tests/golden/bpt_trees_v1.npz')
IMAGES = range(5)


def same_tree(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def golden(kind, index):
    return GOLDEN[f'{kind}_left_{index}'], GOLDEN[f'{kind}_right_{index}']


@pytest.mark.parametrize('index', IMAGES)
def test_defaults_reproduce_the_v1_trees(index):
    image, partition = GOLDEN[f'image_{index}'], GOLDEN[f'partition_{index}']
    assert same_tree(tree(image), golden('plain', index))
    assert same_tree(tree(image, partition), golden('guided', index))
    # Spelling out the v1 defaults changes nothing.
    assert same_tree(tree(image, partition, label_words=1, guide_alpha=2.0, guide_eps=1.0, count_unassigned=True),
                     golden('guided', index))


@pytest.mark.parametrize('index', IMAGES)
def test_alpha_zero_ignores_the_partition(index):  # Proposition 2(i)
    image, partition = GOLDEN[f'image_{index}'], GOLDEN[f'partition_{index}']
    assert same_tree(tree(image, partition, guide_alpha=0.0), golden('plain', index))


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int32, np.int64])
def test_label_map_dtype_does_not_change_the_tree(dtype):
    image, partition = GOLDEN['image_1'], GOLDEN['partition_1']
    assert partition.max() >= 8  # A uint8 shift would have dropped label 8 in v1.
    assert same_tree(tree(image, partition.astype(dtype)), golden('guided', 1))


def test_wider_words_give_the_same_tree_for_small_labels():
    image, partition = GOLDEN['image_2'], GOLDEN['partition_2']
    for words in (2, 4, 'auto'):
        assert same_tree(tree(image, partition, label_words=words), golden('guided', 2))


def test_labels_that_do_not_fit_are_rejected():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    labels = np.zeros((4, 4), dtype=np.int64)
    labels[0, 0] = 64
    with pytest.raises(ValueError, match='does not fit'):
        bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=labels)
    bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=labels, label_words=2)
    labels[0, 0] = 256
    with pytest.raises(ValueError, match='at most 4'):
        bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=labels, label_words='auto')
    labels[0, 0] = -1
    with pytest.raises(ValueError, match='>= 0'):
        bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=labels)
    with pytest.raises(ValueError, match='shape'):
        bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=np.zeros((3, 4), dtype=np.int64))
    for bad in [dict(label_words=3), dict(guide_alpha=-1), dict(guide_alpha='soft'), dict(guide_eps=0)]:
        with pytest.raises(ValueError):
            bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=np.zeros((4, 4), dtype=np.int64), **bad)


def test_255_stripes_fill_the_root_signature():
    width = 255
    image = (np.arange(width)[None, :, None] * np.array([1, 3, 7]) % 256).astype(np.uint8).repeat(3, axis=0)
    labels = np.repeat(np.arange(width, dtype=np.int64)[None, :], 3, axis=0)
    builder = bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=labels, label_words=4)
    builder.compute()
    root = 2 * image.shape[0] * width - 2
    assert builder.label_words == 4
    assert sum(bin(word).count('1') for word in builder.cluster_signature(root)) == 255


def test_high_labels_land_in_the_right_word_and_bit():
    wanted = [63, 64, 127, 128, 254]
    assert [bpt.label_bit_position(label) for label in wanted] == [(0, 63), (1, 0), (1, 63), (2, 0), (3, 62)]
    image = np.zeros((1, len(wanted), 3), dtype=np.uint8)
    builder = bpt.BinaryPartitionTreeBuilder(image=image, prebuilt_partitions=np.array([wanted]), label_words=4)
    for pixel, label in enumerate(wanted):
        word, bit = bpt.label_bit_position(label)
        expected = [0, 0, 0, 0]
        expected[word] = 1 << bit
        assert builder.cluster_signature(pixel) == tuple(expected)


def test_unassigned_label_can_be_left_out_of_the_count():
    image, partition = GOLDEN['image_3'], GOLDEN['partition_3']
    assert (partition == 0).any()
    counted = tree(image, partition)
    uncounted = tree(image, partition, count_unassigned=False)
    assert not same_tree(counted, uncounted)


def test_base_priorities_are_recorded():
    image, partition = GOLDEN['image_0'], GOLDEN['partition_0']
    builder = bpt.BinaryPartitionTreeBuilder(image=shap_bpt.shap_bpt.image_rgb2lab(image), prebuilt_partitions=partition)
    builder.compute()
    assert 0 < builder.min_base_priority <= builder.max_base_priority < np.inf


def test_hard_guidance_keeps_every_label_component_as_one_node():  # Proposition 2(ii)
    from scipy import ndimage as ndi
    for index in IMAGES:
        image, partition = GOLDEN[f'image_{index}'], GOLDEN[f'partition_{index}']
        built = shap_bpt.build_bpt_from_image(image, prebuilt_partitions=partition, guide_alpha='hard')
        unguided = shap_bpt.build_bpt_from_image(image)
        missing_unguided = 0
        for label in np.unique(partition):
            components, count = ndi.label(partition == label, structure=np.ones((3, 3)))  # 8-connected, as the BPT
            for component in range(1, count + 1):
                mask = components == component
                assert shap_bpt.tree_contains_region(built, mask), (index, label, component)
                missing_unguided += not shap_bpt.tree_contains_region(unguided, mask)
        assert missing_unguided > 0  # The property is not trivially true without guidance.


def test_tree_contains_region_basics():
    image = GOLDEN['image_0']
    built = shap_bpt.build_bpt_from_image(image)
    everything = np.ones(image.shape[:2], dtype=bool)
    single = np.zeros(image.shape[:2], dtype=bool)
    single[3, 4] = True
    assert shap_bpt.tree_contains_region(built, everything) and shap_bpt.tree_contains_region(built, single)
    assert not shap_bpt.tree_contains_region(built, np.zeros_like(single))


# ---------------------------------------------------------------- pipeline options (guide_options)

def runner_args(*extra):
    sys.path[:0] = [str(ROOT / 'src/scripts/explanations'), str(ROOT / 'src/scripts')]
    import experiment_common as common
    return common, common.create_parser('OC_ResNet').parse_args(['--budget', '3000', *extra])


def test_default_options_keep_names_folders_and_signatures():
    import argparse
    common, args = runner_args()
    import all_methods
    import guide_options
    assert guide_options.is_default(args) and guide_options.tree_kwargs(args) == {}
    assert guide_options.max_labels(args) == 63
    assert common.run_folder(args).endswith('_B3000')
    assert all_methods.method_names(args)[:3] == ['BPT-3000', 'BPT+SAM_c-3000', 'SAM_c+KernelSHAP-3000']
    old = argparse.Namespace(**{key: value for key, value in vars(args).items() if key not in guide_options.DEFAULTS})
    assert common.run_signature('OC_ResNet', args) == common.run_signature('OC_ResNet', old)
    assert all_methods.comparison_signature(args) == all_methods.comparison_signature(old)


def test_non_default_options_are_named_and_change_the_signature():
    common, args = runner_args('--label-words', '2', '--guide-alpha', '4')
    _, default = runner_args()
    import all_methods
    import guide_options
    assert guide_options.tree_kwargs(args) == {'label_words': 2, 'guide_alpha': 4.0, 'guide_eps': 1.0,
                                               'count_unassigned': True}
    assert guide_options.max_labels(args) == 127
    assert common.run_folder(args).endswith('_B3000_W2_a4_e1')
    names = all_methods.method_names(args)
    assert 'BPT+SAM_c@a4W2-3000' in names and names[1].rsplit('-', 1) == ['BPT+SAM_c@a4W2', '3000']
    assert common.run_signature('OC_ResNet', args) != common.run_signature('OC_ResNet', default)
    _, hard = runner_args('--guide-alpha', 'hard', '--no-count-unassigned', '--label-words', 'auto')
    assert guide_options.method_tag(hard) == '@ahardWautou0' and guide_options.max_labels(hard) == 255
    assert common.run_folder(hard).endswith('_Wauto_ahard_e1_u0')


def test_wider_cap_keeps_more_labels():
    sys.path.insert(0, str(ROOT / 'src/scripts'))
    from utils_sam_2 import cap_partition_labels, sanitize_partitions
    labels = np.repeat(np.arange(200)[None, :], 2, axis=0)
    image_shape = (*labels.shape, 3)
    assert len(np.unique(sanitize_partitions(cap_partition_labels(labels, 63), image_shape, 63))) == 63
    wide = sanitize_partitions(cap_partition_labels(labels, 255), image_shape, 255)
    assert len(np.unique(wide)) == 200 and wide.max() == 199  # Labels above 255 would have wrapped as uint8.
