"""Portable functions from notebooks/pipeline-sam.ipynb.

Definitions only: importing this module never runs notebook installation or inference.
Notebook algorithms are preserved; paths and saved filenames are made portable.
"""
import copy
import time
from pathlib import Path
from io import BytesIO
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm
import imageio.v2 as imageio
import json

def show_all_masks(
    image,
    sam_masks,
    compact_mask,
    title=None,
    alpha=0.45,
    rainbow_name="turbo",
    color_start=0.25
):
    """
    Mostra:
    1. immagine originale;
    2. compact mask;
    3. overlay delle maschere SAM.

    compact_mask viene passata direttamente, quindi non viene ricalcolata.
    """

    if compact_mask is None:
        raise ValueError("compact_mask non può essere None")

    n_masks = len(sam_masks)

    # L'ID massimo reale presente nella compact mask
    max_id = int(np.max(compact_mask))

    cmap, norm = black_rainbow_colormap(
        max_id,
        rainbow_name=rainbow_name,
        start=color_start
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7),
        constrained_layout=True
    )

    ax_img, ax_seg, ax_overlay = axes

    # ----------------------------------------------------------
    # 1. Immagine originale
    # ----------------------------------------------------------
    ax_img.imshow(image)
    ax_img.set_title("Original Image")
    ax_img.axis("off")

    # ----------------------------------------------------------
    # 2. Compact mask
    # ----------------------------------------------------------
    ax_seg.imshow(
        compact_mask,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax_seg.set_title(
        f"Compact Mask\n"
        f"{len(np.unique(compact_mask)) - int(0 in compact_mask)} instances"
    )
    ax_seg.axis("off")

    # ----------------------------------------------------------
    # 3. Overlay
    # ----------------------------------------------------------
    ax_overlay.imshow(image)

    for i, mask_data in enumerate(sam_masks):
        mask = mask_data["segmentation"].astype(bool)

        # Gli ID della compact mask partono da 1.
        mask_id = i + 1

        # Recupera il colore associato esattamente a quell'ID.
        color = cmap(norm(mask_id))

        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[mask, :3] = color[:3]
        overlay[mask, 3] = alpha

        ax_overlay.imshow(overlay)

        ys, xs = np.where(mask)

        if len(xs) > 0:
            ax_overlay.text(
                int(xs.mean()),
                int(ys.mean()),
                str(mask_id),
                color="white",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="black",
                    alpha=0.7,
                    edgecolor="none"
                )
            )

    ax_overlay.set_title(f"Overlay ({n_masks} masks)")
    ax_overlay.axis("off")

    # ----------------------------------------------------------
    # Colorbar
    # ----------------------------------------------------------
    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])

    n_ticks = min(10, max_id)

    if n_ticks > 0:
        ticks = np.unique(
            np.linspace(
                1,
                max_id,
                n_ticks,
                dtype=int
            )
        )

        cbar = fig.colorbar(
            sm,
            ax=ax_overlay,
            ticks=ticks,
            fraction=0.02,
            pad=0.02,
            aspect=30,
            shrink=0.8
        )

        cbar.set_label("Mask ID")

    if title is None:
        title = f"Numero maschere SAM: {n_masks}"

    fig.suptitle(title, fontsize=15)

    plt.show()


def show_compact_overlay(
    image,
    compact_mask,
    alpha=0.45,
    rainbow_name="turbo",
    color_start=0.25,
    title_prefix="FILLED"
):
    if compact_mask is None:
        raise ValueError("compact_mask non può essere None")

    compact_mask = np.asarray(compact_mask)

    ids = np.unique(compact_mask)
    ids = ids[ids != 0]

    n_instances = len(ids)
    max_id = int(compact_mask.max())

    cmap, norm = black_rainbow_colormap(
        max(max_id, 1),
        rainbow_name=rainbow_name,
        start=color_start
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 7),
        constrained_layout=True
    )

    ax_img, ax_seg, ax_overlay = axes

    # ----------------------------------------------------------
    # 1. Immagine originale
    # ----------------------------------------------------------
    ax_img.imshow(image)
    ax_img.set_title("Original Image")
    ax_img.axis("off")

    # ----------------------------------------------------------
    # 2. Compact mask
    # ----------------------------------------------------------
    ax_seg.imshow(
        compact_mask,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    ax_seg.set_title(
        f"{title_prefix} Segmentation\n"
        f"{n_instances} instances"
    )
    ax_seg.axis("off")

    # ----------------------------------------------------------
    # 3. Overlay
    # ----------------------------------------------------------
    ax_overlay.imshow(image)

    for mask_id in ids:
        mask = compact_mask == mask_id

        color = cmap(norm(mask_id))

        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[mask, :3] = color[:3]
        overlay[mask, 3] = alpha

        ax_overlay.imshow(overlay)

        ys, xs = np.where(mask)

        if len(xs) > 0:
            ax_overlay.text(
                int(xs.mean()),
                int(ys.mean()),
                str(mask_id),
                color="white",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="black",
                    alpha=0.7,
                    edgecolor="none"
                )
            )

    ax_overlay.set_title(
        f"Overlay ({n_instances} instances)"
    )
    ax_overlay.axis("off")

    # ----------------------------------------------------------
    # Colorbar
    # ----------------------------------------------------------
    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])

    if n_instances > 0:
        n_ticks = min(10, n_instances)

        tick_positions = np.unique(
            np.linspace(
                0,
                n_instances - 1,
                n_ticks,
                dtype=int
            )
        )

        ticks = ids[tick_positions]

        cbar = fig.colorbar(
            sm,
            ax=ax_overlay,
            ticks=ticks,
            fraction=0.02,
            pad=0.02,
            aspect=30,
            shrink=0.8
        )

        cbar.set_label("Mask ID")

    fig.suptitle(
        f"Numero istanze {title_prefix}: {n_instances}",
        fontsize=15
    )

    plt.show()


def black_rainbow_colormap(
    n: int,
    rainbow_name: str = "turbo",
    start: float = 0.25
) -> tuple[ListedColormap, BoundaryNorm]:

    if n < 1:
        n = 1

    if not 0.0 <= start < 1.0:
        raise ValueError("start deve essere compreso tra 0 e 1")

    rainbow = plt.colormaps[rainbow_name]

    # Esclude la parte iniziale troppo scura della colormap
    rainbow_colors = rainbow(
        np.linspace(start, 1.0, n)
    )

    colors = np.vstack([
        [0.0, 0.0, 0.0, 1.0],
        rainbow_colors
    ])

    cmap = ListedColormap(
        colors,
        name=f"black_{rainbow_name}_{n}"
    )

    boundaries = np.arange(
        -0.5,
        n + 1.5,
        1.0
    )

    norm = BoundaryNorm(
        boundaries,
        cmap.N
    )

    return cmap, norm


def show_saved_masks(
    stem,
    results_dir="results/notebook_pipeline",
    rainbow_name="turbo"
):
    results_dir = Path(results_dir)

    files = [
        ("SAM",      results_dir / f"{stem}_sam.npy"),
        ("Coverage", results_dir / f"{stem}_greedy_selection.npy"),
        ("Compact",  results_dir / f"{stem}_compact_selection.npy"),
        ("Refined",  results_dir / f"{stem}_boosted.npy"),
        ("Filled",   results_dir / f"{stem}_filled.npy"),
    ]

    loaded_masks = []

    # Prima carichiamo tutto per trovare l'ID massimo globale
    for title, path in files:
        if not path.exists():
            loaded_masks.append((title, path, None))
            continue

        mask = np.load(path, allow_pickle=True)

        # SAM e Coverage possono essere liste di dizionari
        if isinstance(mask, np.ndarray) and mask.dtype == object:
            mask = mask.tolist()

            if isinstance(mask, list):
                mask = create_compact_mask(mask)

        mask = np.asarray(mask)

        loaded_masks.append((title, path, mask))

    valid_masks = [
        mask
        for _, _, mask in loaded_masks
        if mask is not None
    ]

    if len(valid_masks) == 0:
        print(f"Nessuna maschera trovata per stem: {stem}")
        return

    max_id = max(
        int(np.max(mask))
        for mask in valid_masks
    )

    cmap, norm = black_rainbow_colormap(
        max_id,
        rainbow_name=rainbow_name
    )

    fig, axes = plt.subplots(
        1,
        len(files),
        figsize=(24, 6),
        constrained_layout=True
    )

    for ax, (title, path, mask) in zip(axes, loaded_masks):
        if mask is None:
            ax.set_title(f"{title}\nNot found")
            ax.axis("off")
            continue

        ax.imshow(
            mask,
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        n_instances = len(np.unique(mask)) - int(0 in mask)

        ax.set_title(
            f"{title}\n"
            f"{n_instances} instances"
        )
        ax.axis("off")

    plt.show()


def create_compact_mask(masks):
    """
    Crea una segmentation mask dando priorità alle maschere più piccole.

    Le maschere vengono ordinate per area decrescente:
    - prima vengono disegnate le più grandi;
    - poi le più piccole le sovrascrivono nelle zone di overlap.
    """

    if len(masks) == 0:
        return None

    sorted_masks = sorted(
        masks,
        key=lambda m: m["area"],
        reverse=True      # area più grande -> più piccola
    )

    h, w = sorted_masks[0]["segmentation"].shape
    compact_mask = np.zeros((h, w), dtype=np.int32)

    for idx, mask_data in enumerate(sorted_masks):
        compact_mask[mask_data["segmentation"]] = idx + 1

    return compact_mask


def compact_mask_to_masks(compact_mask):
    """
    Converte una compact mask in una lista di maschere.

    Ogni elemento della lista ha lo stesso formato utilizzato da SAM,
    contenendo almeno:
        - segmentation
        - area

    Gli ID della compact_mask vengono mantenuti.
    """

    if compact_mask is None:
        return []

    masks = []

    ids = np.unique(compact_mask)
    ids = ids[ids != 0]      # elimina lo sfondo

    for mask_id in ids:
        segmentation = compact_mask == mask_id

        masks.append({
            "id": int(mask_id),
            "segmentation": segmentation,
            "area": int(np.count_nonzero(segmentation))
        })

    return masks


def refiner(segmentationMask, masks):
# https://github.com/99eren99/SegmentBooster/blob/main/segmentBooster/main.py
    masks=sorted(masks, key=(lambda x: x['area']), reverse=True)

    refined_mask = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]), dtype=np.int32)
    for ann in masks:
        m = ann['segmentation']

        unique, counts = np.unique(segmentationMask[m], return_counts=True)

        freqs=(np.asarray((unique, counts)).T)
        freqs=freqs[freqs[:, 1].argsort()]

        mostFrequentID=freqs[-1,0]

        refined_mask[m] = mostFrequentID

    return refined_mask


def expand_matrix(mat, delta=1, sel_threshold=3):
    height, width = mat.shape
    mat2 = mat.copy()

    for x in range(width):
        for y in range(height):
            p0 = mat[y, x]
            if p0 == 0: # background
                p1, p2 = 0, 0
                c1, c2 = 0, 0

                for dx in range(max(0, x-delta), min(width, x+delta+1)):
                    for dy in range(max(0, y-delta), min(height, y+delta+1)):
                        if dx!=x and dy!=y:
                            p = mat[dy, dx]
                            if p != 0:
                                if p1==0:
                                    p1, c1 = p, 1
                                else:
                                    if p1==p:
                                        c1 += 1
                                    else:
                                        if p2==0:
                                            p2, c2 = p, 1
                                        else:
                                            if p2==p:
                                                c2 += 1
                if p1!=0 and c1>sel_threshold:
                    mat2[y, x] = p1
                elif p2!=0 and c2>sel_threshold:
                    mat2[y, x] = p2
    return mat2


def partition_unassigned_by_nearest_cluster(
    labels: np.ndarray,
    max_distance: float = None,
    fill_mask: np.ndarray = None,
    unassigned_label: int = 0,
    keep_ties_unassigned: bool = False,
    spacing: tuple[float, float] = None,
) -> np.ndarray:
    labels = np.asarray(labels)

    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional array")

    target = labels == unassigned_label

    if fill_mask is not None:
        fill_mask = np.asarray(fill_mask, dtype=bool)

        if fill_mask.shape != labels.shape:
            raise ValueError("fill_mask must have the same shape as labels")

        target &= fill_mask

    cluster_ids = np.unique(labels[labels != unassigned_label])

    if cluster_ids.size == 0 or not np.any(target):
        return labels.copy()

    distances = np.stack([
        ndi.distance_transform_edt(
            labels != cluster_id,
            sampling=spacing,
        )
        for cluster_id in cluster_ids
    ])

    winner_index = np.argmin(distances, axis=0)
    best_distance = np.take_along_axis(distances, winner_index[None, ...], axis=0)[0]

    if cluster_ids.size >= 2:
        second_distance = np.partition(distances, kth=1, axis=0)[1]

        tied = np.isclose(
            best_distance,
            second_distance,
            rtol=0.0,
            atol=1e-12,
        )
    else:
        tied = np.zeros(labels.shape, dtype=bool)

    if max_distance is not None:
        target &= best_distance <= max_distance

    if keep_ties_unassigned:
        target &= ~tied

    result = labels.copy()
    result[target] = cluster_ids[winner_index[target]]

    return result


def select_masks_by_unpicked_area(
    masks,
    n_masks_to_select,
    iou_key="predicted_iou",
    create_gif=False,
    gif_path="results/notebook_pipeline/coverage/coverage_progress.gif",
    gif_fps=4,
    frame_every=1,
    save_step_images=False,
    step_images_dir="results/notebook_pipeline/coverage"
):
    """
    Seleziona iterativamente le maschere usando:

        new_score = area_non_ancora_coperta * predicted_iou

    dove:

        unpicked_mask = mask & (~picked_pixels)

        unpicked_area = numero di pixel True in unpicked_mask

    Se create_gif=True, ogni frame mostra:

    1. coverage prima dell'inserimento;
    2. maschera completa selezionata;
    3. andamento dell'area della unpicked_mask delle maschere
       selezionate fino allo step corrente.

    Se save_step_images=True, salva per ogni step:

    - coverage prima dell'aggiunta;
    - maschera completa scelta;
    - unpicked_mask, cioè i soli pixel nuovi;
    - coverage dopo l'aggiunta;
    - grafico dell'area della unpicked_mask.

    Returns
    -------
    coverage_masks : list[dict]
        Maschere selezionate.

    picked_pixels : np.ndarray
        Maschera booleana contenente la coverage finale.

    selection_history : list[dict]
        Statistiche di ogni selezione.

    saved_gif_path : pathlib.Path | None
        Percorso della GIF, oppure None.
    """

    # ----------------------------------------------------------
    # Controlli iniziali
    # ----------------------------------------------------------
    if len(masks) == 0:
        return [], None, [], None

    if frame_every < 1:
        raise ValueError(
            "frame_every deve essere almeno 1"
        )

    if n_masks_to_select < 1:
        raise ValueError(
            "n_masks_to_select deve essere almeno 1"
        )

    if "segmentation" not in masks[0]:
        raise KeyError(
            "Ogni maschera deve contenere la chiave "
            "'segmentation'"
        )

    if iou_key not in masks[0]:
        raise KeyError(
            f"Ogni maschera deve contenere la chiave "
            f"'{iou_key}'"
        )

    h, w = masks[0]["segmentation"].shape

    # False = pixel non ancora coperto
    # True = pixel già coperto
    picked_pixels = np.zeros(
        (h, w),
        dtype=bool
    )

    # Salva anche l'indice originale di ogni maschera
    remaining_masks = list(
        enumerate(masks)
    )

    coverage_masks = []
    selection_history = []

    max_selections = min(
        n_masks_to_select,
        len(masks)
    )

    frames = []

    # ----------------------------------------------------------
    # Dati per il grafico della unpicked area
    # ----------------------------------------------------------
    unpicked_steps = []
    unpicked_areas = []

    # ----------------------------------------------------------
    # Cartella per le immagini dei singoli step
    # ----------------------------------------------------------
    step_dir = None

    if save_step_images:
        step_dir = Path(step_images_dir)

        step_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ----------------------------------------------------------
    # Selezione greedy
    # ----------------------------------------------------------
    for selection_step in range(max_selections):

        best_position = None
        best_score = -1.0
        best_unpicked_area = 0
        best_unpicked_mask = None

        # ------------------------------------------------------
        # Cerca la maschera con lo score migliore
        # ------------------------------------------------------
        for position, (
            original_id,
            mask_data
        ) in enumerate(remaining_masks):

            if "segmentation" not in mask_data:
                raise KeyError(
                    "Ogni maschera deve contenere la chiave "
                    "'segmentation'"
                )

            if iou_key not in mask_data:
                raise KeyError(
                    f"Ogni maschera deve contenere la chiave "
                    f"'{iou_key}'"
                )

            mask = np.asarray(
                mask_data["segmentation"],
                dtype=bool
            )

            if mask.shape != (h, w):
                raise ValueError(
                    "Tutte le maschere devono avere "
                    "la stessa dimensione"
                )

            # Pixel della maschera non ancora coperti
            unpicked_mask = (
                mask & (~picked_pixels)
            )

            # Numero di nuovi pixel
            unpicked_area = int(
                np.count_nonzero(
                    unpicked_mask
                )
            )

            iou = float(
                mask_data[iou_key]
            )

            new_score = (
                unpicked_area * iou
            )

            if new_score > best_score:
                best_position = position
                best_score = new_score
                best_unpicked_area = unpicked_area
                best_unpicked_mask = unpicked_mask

        # ------------------------------------------------------
        # Nessuna maschera aggiunge nuovi pixel
        # ------------------------------------------------------
        if (
            best_position is None
            or best_unpicked_area == 0
        ):
            break

        # Rimuove la maschera scelta da quelle disponibili
        original_id, best_mask = remaining_masks.pop(
            best_position
        )

        current_mask = np.asarray(
            best_mask["segmentation"],
            dtype=bool
        )

        current_iou = float(
            best_mask[iou_key]
        )

        current_step = (
            selection_step + 1
        )

        # Coverage prima dell'aggiunta
        coverage_before = picked_pixels.copy()

        coverage_masks.append(
            best_mask
        )

        # Aggiunge solo i pixel non ancora coperti
        picked_pixels |= best_unpicked_mask

        # Coverage dopo l'aggiunta
        coverage_after = picked_pixels.copy()

        total_picked_pixels = int(
            np.count_nonzero(
                picked_pixels
            )
        )

        coverage_percentage = (
            total_picked_pixels
            / picked_pixels.size
        ) * 100.0

        # ------------------------------------------------------
        # Aggiorna i dati del grafico
        # ------------------------------------------------------
        unpicked_steps.append(
            current_step
        )

        unpicked_areas.append(
            best_unpicked_area
        )

        # ------------------------------------------------------
        # Salva le statistiche dello step
        # ------------------------------------------------------
        selection_history.append({
            "selection_order": current_step,
            "original_mask_id": int(original_id),
            "predicted_iou": current_iou,
            "unpicked_area": int(
                best_unpicked_area
            ),
            "new_score": float(
                best_score
            ),
            "total_picked_pixels": (
                total_picked_pixels
            ),
            "coverage_percentage": float(
                coverage_percentage
            )
        })

        # ------------------------------------------------------
        # Salvataggio immagini singole
        # ------------------------------------------------------
        if save_step_images:
            step_name = (
                f"step_{current_step:04d}"
            )

            plt.imsave(
                step_dir
                / f"{step_name}_coverage_before.png",
                coverage_before.astype(
                    np.uint8
                ),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            plt.imsave(
                step_dir
                / f"{step_name}_selected_mask.png",
                current_mask.astype(
                    np.uint8
                ),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            plt.imsave(
                step_dir
                / f"{step_name}_unpicked_mask.png",
                best_unpicked_mask.astype(
                    np.uint8
                ),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            plt.imsave(
                step_dir
                / f"{step_name}_coverage_after.png",
                coverage_after.astype(
                    np.uint8
                ),
                cmap="gray",
                vmin=0,
                vmax=1
            )

            # --------------------------------------------------
            # Salva il grafico della unpicked area
            # --------------------------------------------------
            fig_plot, ax_plot = plt.subplots(
                figsize=(7, 5)
            )

            ax_plot.plot(
                unpicked_steps,
                unpicked_areas,
                marker="o"
            )

            # Evidenzia il punto corrente
            ax_plot.scatter(
                [current_step],
                [best_unpicked_area],
                s=70,
                zorder=3
            )

            ax_plot.annotate(
                f"{best_unpicked_area}",
                xy=(
                    current_step,
                    best_unpicked_area
                ),
                xytext=(8, 8),
                textcoords="offset points"
            )

            ax_plot.set_title(
                "Area della unpicked mask"
            )

            ax_plot.set_xlabel(
                "Numero di maschere selezionate"
            )

            ax_plot.set_ylabel(
                "Nuovi pixel aggiunti"
            )

            ax_plot.set_xlim(
                1,
                max_selections
            )

            # L'area massima possibile è il numero
            # totale di pixel dell'immagine
            ax_plot.set_ylim(
                0,
                h * w
            )

            ax_plot.grid(
                True,
                alpha=0.3
            )

            fig_plot.tight_layout()

            fig_plot.savefig(
                step_dir
                / f"{step_name}_unpicked_area_plot.png",
                dpi=120,
                bbox_inches="tight"
            )

            plt.close(
                fig_plot
            )

        # ------------------------------------------------------
        # Creazione frame GIF
        # ------------------------------------------------------
        should_create_frame = (
            create_gif
            and (
                selection_step % frame_every == 0
                or selection_step == max_selections - 1
            )
        )

        if should_create_frame:
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5),
                constrained_layout=True
            )

            # --------------------------------------------------
            # 1. Coverage prima dell'inserimento
            # --------------------------------------------------
            axes[0].imshow(
                coverage_before,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[0].set_title(
                "Coverage prima dell'inserimento\n"
                f"Step {current_step}"
            )

            axes[0].axis("off")

            # --------------------------------------------------
            # 2. Maschera completa selezionata
            # --------------------------------------------------
            axes[1].imshow(
                current_mask,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[1].set_title(
                "Maschera da aggiungere\n"
                f"ID originale {original_id}"
            )

            axes[1].axis("off")

            # --------------------------------------------------
            # 3. Andamento della unpicked area
            # --------------------------------------------------
            axes[2].plot(
                unpicked_steps,
                unpicked_areas,
                marker="o",
                linewidth=2
            )
            
            axes[2].scatter(
                current_step,
                best_unpicked_area,
                color="red",
                s=80,
                zorder=3
            )
            
            axes[2].annotate(
                f"{best_unpicked_area}",
                xy=(current_step, best_unpicked_area),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=9
            )
            
            axes[2].set_title("Area della unpicked mask")
            
            axes[2].set_xlabel("Maschere selezionate")
            axes[2].set_ylabel("Nuovi pixel")
            
            axes[2].set_xlim(1, max_selections)
            
            # -----------------------------
            # Zoom automatico dell'asse Y
            # -----------------------------
            if len(unpicked_areas) > 5:
                ymax = np.percentile(unpicked_areas, 95)
            else:
                ymax = max(unpicked_areas)
            
            ymax = max(ymax, best_unpicked_area)
            
            axes[2].set_ylim(
                0,
                ymax * 1.10
            )
            
            axes[2].grid(True, alpha=0.3)

            fig.suptitle(
                f"IoU={current_iou:.3f} | "
                f"Nuovi pixel={best_unpicked_area} | "
                f"Score={best_score:.2f} | "
                f"Coverage dopo={coverage_percentage:.2f}%"
            )

            buffer = BytesIO()

            fig.savefig(
                buffer,
                format="png",
                dpi=120
            )

            plt.close(
                fig
            )

            buffer.seek(0)

            frame = imageio.imread(
                buffer
            )

            frames.append(
                frame
            )

            buffer.close()

    # ----------------------------------------------------------
    # Salvataggio GIF
    # ----------------------------------------------------------
    saved_gif_path = None

    if create_gif and len(frames) > 0:
        saved_gif_path = Path(
            gif_path
        )

        saved_gif_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        imageio.mimsave(
            saved_gif_path,
            frames,
            duration=1000 / gif_fps,
            loop=0
        )

        print(
            f"GIF salvata in: {saved_gif_path}"
        )

    if save_step_images:
        print(
            "Immagini dei singoli step salvate in: "
            f"{step_dir}"
        )

    return (
        coverage_masks,
        picked_pixels,
        selection_history,
        saved_gif_path
    )


def compute_coverage(mask):
    """
    Calcola la copertura di una maschera di segmentazione.

    Returns
    -------
    non_zero_pixels : int
        Numero di pixel appartenenti ad almeno una istanza.

    total_pixels : int
        Numero totale di pixel dell'immagine.

    coverage : float
        Percentuale di copertura [0,100].
    """
    if(mask) is None:
        return None
    non_zero_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    coverage = 100 * non_zero_pixels / total_pixels

    return non_zero_pixels, total_pixels, coverage


def save_results(
    image_name,
    sam_masks,
    coverage_masks,
    compact_mask,
    filled_mask,
    params,
    image_result,
    refined_mask,
    fillMask=False,
    refineMask=False,
    output_dir="results/notebook_pipeline"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem

    json_path = output_dir / f"{stem}_results.json"
    sam_path = output_dir / f"{stem}_sam.npy"
    coverage_path = output_dir / f"{stem}_greedy_selection.npy"
    compact_path = output_dir / f"{stem}_compact_selection.npy"

    filled_path = None
    refined_path = None

    # ----------------------------------------------------------
    # SAM compact ottenuta da tutte le maschere SAM
    # ----------------------------------------------------------
    sam_compact = create_compact_mask(sam_masks)

    np.save(
        sam_path,
        np.asarray(sam_compact)
    )

    # ----------------------------------------------------------
    # Coverage masks
    # ----------------------------------------------------------
    np.save(
        coverage_path,
        np.asarray(
            coverage_masks,
            dtype=object
        ),
        allow_pickle=True
    )

    # ----------------------------------------------------------
    # Compact mask
    # ----------------------------------------------------------
    if compact_mask is not None:
        np.save(
            compact_path,
            np.asarray(compact_mask)
        )
    else:
        compact_path = None

    # ----------------------------------------------------------
    # Filled mask opzionale
    # ----------------------------------------------------------
    if filled_mask is not None:

        if fillMask and refineMask:
            filled_path = output_dir / f"{stem}_boosted_filled.npy"
        else:
            filled_path = output_dir / f"{stem}_filled.npy"
        
        np.save(
            filled_path,
            np.asarray(filled_mask)
        )

    # ----------------------------------------------------------
    # Refined mask opzionale
    # ----------------------------------------------------------
    if refined_mask is not None:
        refined_path = output_dir / f"{stem}_boosted.npy"

        np.save(
            refined_path,
            np.asarray(refined_mask)
        )

    # ----------------------------------------------------------
    # Aggiornamento path nel JSON
    # ----------------------------------------------------------
    image_result.update({
        "json_path": str(json_path),

        "sam_path": str(sam_path),

        "coverage_path": str(coverage_path),

        "compact_path": (
            str(compact_path)
            if compact_path is not None
            else None
        ),

        "filled_path": (
            str(filled_path)
            if filled_path is not None
            else None
        ),

        "refined_path": (
            str(refined_path)
            if refined_path is not None
            else None
        ),
    })

    # ----------------------------------------------------------
    # Salvataggio JSON
    # ----------------------------------------------------------
    with json_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            image_result,
            f,
            indent=4,
            ensure_ascii=False,
            default=str
        )

    return (
        json_path,
        sam_path,
        coverage_path,
        compact_path,
        filled_path,
        refined_path,
    )


def generate_at_least_n_masks(
    image,
    model,
    base_params,
    min_masks=400,
    min_pred_iou_thresh=0.10,
    min_stability_score_thresh=0.10,
    step=0.2,
    refineMask=False,
    fillMask=False,
    showPlots=False,
    max_steps=3,
    create_gif=False,
    save_step_images=False    
):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    params = copy.deepcopy(base_params)

    total_start = time.time()
    steps = 1
    params_history = []

    while True:
        generator = SAM2AutomaticMaskGenerator(
            model,
            **params
        )

        start_sam = time.time()
        sam_masks = generator.generate(image)
        elapsed_sam = time.time() - start_sam

        print(
            f"Step={steps}/{max_steps} | "
            f"pred_iou_thresh={params['pred_iou_thresh']:.2f} | "
            f"stability_score_thresh={params['stability_score_thresh']:.2f} | "
            f"maschere={len(sam_masks)} | "
            f"tempo={elapsed_sam:.2f}s"
        )

        params_history.append(
            f"iou={params['pred_iou_thresh']:.2f}-"
            f"stab={params['stability_score_thresh']:.2f}-"
            f"masks={len(sam_masks)}-"
            f"sec={elapsed_sam:.2f}"
        )

        enough_masks = len(sam_masks) >= min_masks
        max_steps_reached = steps >= max_steps

        thresholds_reached = (
            params["pred_iou_thresh"] <= min_pred_iou_thresh
            and
            params["stability_score_thresh"]
            <= min_stability_score_thresh
        )

        if enough_masks or max_steps_reached or thresholds_reached:

            if enough_masks:
                print(
                    f"Numero minimo raggiunto: "
                    f"{len(sam_masks)} >= {min_masks}"
                )

            if max_steps_reached and not enough_masks:
                print(f"Max steps raggiunto: {steps}")

            if thresholds_reached and not enough_masks:
                print(
                    "Soglie minime raggiunte: "
                    f"IoU={params['pred_iou_thresh']:.2f}, "
                    f"stability="
                    f"{params['stability_score_thresh']:.2f}"
                )
                
            # ------------------------------------------------------
            # Selezione greedy
            # ------------------------------------------------------
            start_coverage = time.time()

            (
                coverage_masks,
                picked_pixels,
                selection_history,
                gif_path
            ) = select_masks_by_unpicked_area(
                masks=sam_masks,
                n_masks_to_select=400,
                create_gif=create_gif,
                gif_path="results/notebook_pipeline/coverage/coverage_progress.gif",
                gif_fps=1,
                frame_every=1,
                save_step_images=save_step_images,
                step_images_dir="results/notebook_pipeline/coverage"
            )
            

            elapsed_coverage = time.time() - start_coverage

            # ------------------------------------------------------
            # Compact mask
            # ------------------------------------------------------
            start_compact = time.time()

            compact_mask = create_compact_mask(
                coverage_masks
            )

            elapsed_compact = time.time() - start_compact

            # Plot maschere SAM opzionale
            if showPlots:
                show_all_masks(
                    image=image,
                    sam_masks=sam_masks,
                    compact_mask=compact_mask,
                    rainbow_name="turbo",
                    color_start=0.30
                )

            # Valori predefiniti se le fasi non vengono eseguite
            refined_mask = None
            filled_mask = None

            elapsed_refine = 0.0
            elapsed_filler = 0.0

            # ------------------------------------------------------
            # Refinement opzionale
            # ------------------------------------------------------
            if refineMask:

                start_refine = time.time()

                refined_mask = refiner(
                    compact_mask,
                    coverage_masks
                )

                elapsed_refine = time.time() - start_refine

            # ------------------------------------------------------
            # Filling opzionale
            # ------------------------------------------------------
            if fillMask:

                start_filler = time.time()

                fill_input = (
                    refined_mask
                    if refined_mask is not None
                    else compact_mask
                )

                filled_mask = partition_unassigned_by_nearest_cluster(
                    fill_input,
                    keep_ties_unassigned=False,
                    max_distance=3
                )

                elapsed_filler = time.time() - start_filler

                # Plot filled mask opzionale
                if showPlots:
                    show_compact_overlay(
                        image=image,
                        compact_mask=filled_mask,
                        rainbow_name="turbo",
                        color_start=0.30,
                        title_prefix="FILLED"
                    )

            elif refined_mask is not None:

                # Plot refined mask opzionale
                if showPlots:
                    show_compact_overlay(
                        image=image,
                        compact_mask=refined_mask,
                        rainbow_name="turbo",
                        color_start=0.30,
                        title_prefix="FILLED"
                    )

            elapsed_configuration = time.time() - total_start

            return (
                sam_masks,
                coverage_masks,
                compact_mask,
                filled_mask,
                refined_mask,
                params,
                elapsed_configuration,
                elapsed_refine,
                elapsed_filler,
                elapsed_compact,
                elapsed_coverage,
                elapsed_sam,
                steps,
                params_history,
            )

        # ----------------------------------------------------------
        # Aggiornamento soglie
        # ----------------------------------------------------------
        old_iou = params["pred_iou_thresh"]
        old_stability = params["stability_score_thresh"]

        params["pred_iou_thresh"] = max(
            min_pred_iou_thresh,
            old_iou - step
        )

        params["stability_score_thresh"] = max(
            min_stability_score_thresh,
            old_stability - step
        )

        if (
            params["pred_iou_thresh"] == old_iou
            and
            params["stability_score_thresh"] == old_stability
        ):
            print("I parametri non possono essere ridotti ulteriormente.")

        steps += 1
