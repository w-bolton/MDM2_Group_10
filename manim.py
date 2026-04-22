from manim import *
from manim.utils.color import BLACK, WHITE
from manim import config
import os

# Black background
config.background_color = "#000000" #black

class ObservedRowTransition(Scene):
    def construct(self):
        # --------------------------------------------------
        # 1) Axes
        # --------------------------------------------------
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 1, 0.5],
            x_length=10,
            y_length=5.6,
            axis_config={
                "color": BLACK,
                "stroke_width": 2,
                "include_ticks": True,
                "font_size": 28,
            },
            tips=False,
        )

        #x_label = Text("x", color=WHITE, font_size=28).next_to(axes.x_axis, DOWN, buff=0.2)
        #y_label = Text("z", color=WHITE, font_size=28).next_to(axes.y_axis, LEFT, buff=0.2)

        title = Text("Observed", color=WHITE, font_size=34).next_to(axes, UP, buff=0.25)

        #self.play(Create(axes), Write(x_label), Write(y_label), FadeIn(title))
        self.play(Write(title))
        self.wait(0.3)

        # --------------------------------------------------
        # 2) File list
        # --------------------------------------------------
        image_files = [
            "observed0_nt.png",
            "observed1_nt.png",
            "observed2_nt.png",
            "observed3_nt.png",
            "observed4_nt.png",
        ]

        # Check files exist
        for f in image_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Could not find image file: {f}")

        # --------------------------------------------------
        # 3) Helper to place image exactly over axes region
        # --------------------------------------------------
        def make_fitted_image(filename):
            img = ImageMobject(filename)

            # Fit inside plotting area
            # Slightly smaller than axes box so borders stay visible
            img.set_width(axes.width * 0.985)
            img.set_height(axes.height * 0.985)

            # Center the image on the axes
            img.move_to(axes.get_center())

            return img

        # --------------------------------------------------
        # 4) First frame
        # --------------------------------------------------
        current_img = make_fitted_image(image_files[0])

        # Optional time label
        time_label = Text("t = 0", color=WHITE, font_size=28)
        time_label.next_to(title, RIGHT, buff=0.4)

        self.play(FadeIn(current_img), FadeIn(time_label))
        self.wait(0.6)

        # --------------------------------------------------
        # 5) Smooth transitions between images
        # --------------------------------------------------
        for i in range(1, len(image_files)):
            next_img = make_fitted_image(image_files[i])
            next_time_label = Text(f"t = {i}", color=WHITE, font_size=28).move_to(time_label)

            # Smooth crossfade
            self.play(
                FadeIn(next_img),
                FadeOut(current_img),
                Transform(time_label, next_time_label),
                run_time=0.8,
            )

            current_img = next_img
            self.wait(0.25)

        self.wait(1.0)


from manim import *

config.background_color = "#000000"


class StatisticalInferenceEquations(Scene):
    def construct(self):
        # Title
        title = Text(
            "Statistical Inference",
            font_size=34,
            color=WHITE
        ).to_edge(UP, buff=0.35)

        self.play(Write(title))
        self.wait(0.3)

        # Main loss / log-posterior style equation
        eq1 = MathTex(
            r"\mathcal{L}(\rho)="
            r"-\frac{1}{2\sigma_{\mathrm{obs}}^{2}}\sum_{t=0}^{T}\left\|\mathbf{b}_t - O(\rho_t)\right\|^{2}"
            r"-\frac{1}{2\sigma_{\mathrm{dyn}}^{2}}\sum_{t=0}^{T-1}\left\|\rho_{t+1}-F(\rho_t)\right\|^{2}"
            r"-\frac{1}{2\sigma_{\rho}^{2}}\left\|\rho_0-\rho_{\mathrm{prior}}\right\|^{2}",
            font_size=34,
            color=WHITE
        )

        eq1_box = SurroundingRectangle(
            eq1,
            buff=0.35,
            color=BLACK,
            fill_color=BLACK,
            fill_opacity=1,
            stroke_width=0
        )

        eq1_group = VGroup(eq1_box, eq1).move_to([0, 2.1, 0])

        self.play(FadeIn(eq1_box), Write(eq1))
        self.wait(0.5)

        # Labels underneath the three terms
        obs_label = Tex(
            r"observation likelihood",
            font_size=22,
            color=WHITE
        )
        dyn_label = Tex(
            r"dynamical consistency prior",
            font_size=22,
            color=WHITE
        )
        ic_label = Tex(
            r"initial condition prior",
            font_size=22,
            color=WHITE
        )

        # Position these approximately under the three parts
        obs_label.move_to(eq1.get_center() + LEFT * 3.65 + DOWN * 0.9)
        dyn_label.move_to(eq1.get_center() + RIGHT * 0.15 + DOWN * 0.9)
        ic_label.move_to(eq1.get_center() + RIGHT * 4.0 + DOWN * 0.9)

        obs_box = SurroundingRectangle(obs_label, buff=0.15, color=BLACK, fill_color=BLACK, fill_opacity=1, stroke_width=0)
        dyn_box = SurroundingRectangle(dyn_label, buff=0.15, color=BLACK, fill_color=BLACK, fill_opacity=1, stroke_width=0)
        ic_box = SurroundingRectangle(ic_label, buff=0.15, color=BLACK, fill_color=BLACK, fill_opacity=1, stroke_width=0)

        self.play(
            FadeIn(obs_box), Write(obs_label),
            FadeIn(dyn_box), Write(dyn_label),
            FadeIn(ic_box), Write(ic_label),
            run_time=1.4
        )
        self.wait(0.5)

        # V(rho) = -L(rho)
        eq2 = MathTex(
            r"V(\rho)=-\mathcal{L}(\rho)",
            font_size=34,
            color=WHITE
        )
        eq2_box = SurroundingRectangle(
            eq2, buff=0.25,
            color=BLACK, fill_color=BLACK, fill_opacity=1, stroke_width=0
        )
        eq2_group = VGroup(eq2_box, eq2).move_to([0.55, 0.85, 0])

        self.play(FadeIn(eq2_box), Write(eq2))
        self.wait(0.4)

        # T(r) = 1/2 r^T r
        eq3 = MathTex(
            r"T(r)=\frac{1}{2}r^{T}r",
            font_size=34,
            color=WHITE
        )
        eq3_box = SurroundingRectangle(
            eq3, buff=0.25,
            color=BLACK, fill_color=BLACK, fill_opacity=1, stroke_width=0
        )
        eq3_group = VGroup(eq3_box, eq3).move_to([0.2, -0.2, 0])

        self.play(FadeIn(eq3_box), Write(eq3))
        self.wait(0.4)

        # H(rho, r) = -L(rho) - 1/2 r^T r
        eq4 = MathTex(
            r"H(\rho,r)=-\mathcal{L}(\rho)-\frac{1}{2}r^{T}r",
            font_size=34,
            color=WHITE
        )
        eq4_box = SurroundingRectangle(
            eq4, buff=0.25,
            color=BLACK, fill_color=BLACK, fill_opacity=1, stroke_width=0
        )
        eq4_group = VGroup(eq4_box, eq4).move_to([0.1, -1.25, 0])

        self.play(FadeIn(eq4_box), Write(eq4))
        self.wait(0.4)

        # alpha acceptance equation
        eq5 = MathTex(
            r"\alpha=\min\left(1,\frac{P(\hat{\rho},\hat{r})}{P(\rho,r)}\right)",
            font_size=34,
            color=WHITE
        )
        eq5_box = SurroundingRectangle(
            eq5, buff=0.25,
            color=BLACK, fill_color=BLACK, fill_opacity=1, stroke_width=0
        )
        eq5_group = VGroup(eq5_box, eq5).move_to([0.2, -2.45, 0])

        self.play(FadeIn(eq5_box), Write(eq5))
        self.wait(1.5)

class StatsData(Scene):
    def construct(self):
        # -----------------------------
        # File names: replace these
        # -----------------------------
        fig1_name = "RMSE.png"
        fig2_name = "credibility.png"

        # -----------------------------
        # First figure: full screen
        # -----------------------------
        fig1 = ImageMobject(fig1_name)
        fig1.scale_to_fit_width(config.frame_width * 0.9)
        fig1.scale_to_fit_height(config.frame_height * 0.8)
        fig1.move_to(ORIGIN)

        self.wait(2)
        self.play(FadeIn(fig1), run_time=1.2)
        self.wait(11)

        # -----------------------------
        # Move first figure to left
        # -----------------------------
        fig1.generate_target()
        fig1.target.scale_to_fit_width(config.frame_width * 0.42)
        fig1.target.scale_to_fit_height(config.frame_height * 0.5)
        fig1.target.move_to(LEFT * 3 + UP * 0.4)

        self.play(MoveToTarget(fig1), run_time=1.2)
        self.wait(0.3)

        # -----------------------------
        # RMSE text on right
        # -----------------------------
        rmse_prior = Text("RMSE prior = 0.1231", font_size=34, color=WHITE)
        rmse_hmc = Text("RMSE hmc = 0.0510", font_size=34, color=WHITE)

        rmse_group = VGroup(rmse_prior, rmse_hmc).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        rmse_group.move_to(RIGHT * 3 + UP * 0.3)

        self.play(Write(rmse_group), run_time=1.0)
        self.wait(5)

        # -----------------------------
        # Move text under figure 1
        # -----------------------------
        rmse_group.generate_target()
        rmse_group.target.scale(0.85)
        rmse_group.target.next_to(fig1, DOWN, buff=0.4)

        self.play(MoveToTarget(rmse_group), run_time=1.0)
        self.wait(0.4)

        # -----------------------------
        # Fade in second figure on right
        # -----------------------------
        fig2 = ImageMobject(fig2_name)
        fig2.scale_to_fit_width(config.frame_width * 0.42)
        fig2.scale_to_fit_height(config.frame_height * 0.5)
        fig2.move_to(RIGHT * 3 + UP * 0.4)
        footnote = Paragraph("credibility plot: shows the amount",
        " of data that falls into the reigon",
        " HMC predicts 95% of data is in",line_spacing=0.5 ,font_size=12)
        footnote.set(width = fig2.width * 0.9)
        footnote.next_to(fig2, DOWN, buff=0.3)

        self.play(FadeIn(fig2), FadeIn(footnote), run_time=1.2)
        self.wait(1.5)


from manim import *
import os

config.background_color = "#000000"


class DualRowTransition(Scene):
    def construct(self):
        # --------------------------------------------
        # Layout positions
        # --------------------------------------------
        left_center = LEFT * 3.2
        right_center = RIGHT * 3.2

        img_width = 5.2
        img_height = 3.8
        """
        # --------------------------------------------
        # Titles
        # --------------------------------------------
        left_img = make_image(left_files[0], left_center)
        right_img = make_image(right_files[0], right_center)
        left_title = Text("Observed", color=WHITE, font_size=32)
        right_title = Text("Predicted", color=WHITE, font_size=32)
        left_title.next_to(left_img, UP, buff=0.3)
        right_title.next_to(right_img, UP, buff=0.3)

        self.play(Write(left_title), Write(right_title))
        self.wait(0.3)
        """

        # --------------------------------------------
        # File lists: replace with your actual files
        # --------------------------------------------
        left_files = [
            "observed0_nt.png",
            "observed1_nt.png",
            "observed2_nt.png",
            "observed3_nt.png",
            "observed4_nt.png",
        ]

        right_files = [
            "predicted0_nt.png",
            "predicted1_nt.png",
            "predicted2_nt.png",
            "predicted3_nt.png",
            "predicted4_nt.png",
        ]

        # Check files exist
        for f in left_files + right_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Could not find image file: {f}")

        if len(left_files) != len(right_files):
            raise ValueError("Both image lists must have the same number of frames.")


        #self.play(Write(left_title), Write(right_title))
        #self.wait(0.3)

        # --------------------------------------------
        # Helper function
        # --------------------------------------------
        img_width = 3.8   # smaller than before
        img_height = 2.7
        def make_image(filename, center):
            img = ImageMobject(filename)
            img.set_width(img_width)
            img.set_height(img_height)
            img.move_to(center)
            return img



        # --------------------------------------------
        # Initial images
        # --------------------------------------------
        left_img = make_image(left_files[0], left_center)
        right_img = make_image(right_files[0], right_center)

        left_title = Text("Observed", color=WHITE, font_size=32)
        right_title = Text("Predicted", color=WHITE, font_size=32)
        left_title.next_to(left_img, UP, buff=0.3)
        right_title.next_to(right_img, UP, buff=0.3)

        self.wait(9)
        self.play(Write(left_title))
        self.wait(0.3)

        left_time = Text("t = 0", color=WHITE, font_size=26).next_to(left_title, RIGHT, buff=0.3)
        right_time = Text("t = 0", color=WHITE, font_size=26).next_to(right_title, RIGHT, buff=0.3)

        # Fade in first image only
        self.play(FadeIn(left_img), FadeIn(left_time), run_time=1.0)
        self.wait(3.2)

        # Fade in second image only
        self.play(Write(right_title))
        self.wait(0.3)
        self.play(FadeIn(right_img), FadeIn(right_time), run_time=1.0)
        self.wait(3)

        # --------------------------------------------
        # Play both sequences simultaneously
        # --------------------------------------------
        current_left = left_img
        current_right = right_img
        #self.wait(4)
        for i in range(1, len(left_files)):
            next_left = make_image(left_files[i], left_center)
            next_right = make_image(right_files[i], right_center)

            next_left_time = Text(f"t = {i}", color=WHITE, font_size=26).move_to(left_time)
            next_right_time = Text(f"t = {i}", color=WHITE, font_size=26).move_to(right_time)

            self.play(
                FadeIn(next_left),
                FadeOut(current_left),
                FadeIn(next_right),
                FadeOut(current_right),
                Transform(left_time, next_left_time),
                Transform(right_time, next_right_time),
                run_time=0.8,
            )

            current_left = next_left
            current_right = next_right
            self.wait(0.25)

        self.wait(1.0)


class DensityTransition(Scene):
    def construct(self):
        # --------------------------------------------
        # Layout positions
        # --------------------------------------------
        left_center = LEFT * 3.2
        right_center = RIGHT * 3.2

        img_width = 5.2
        img_height = 3.8
        """
        # --------------------------------------------
        # Titles
        # --------------------------------------------
        left_img = make_image(left_files[0], left_center)
        right_img = make_image(right_files[0], right_center)
        left_title = Text("Observed", color=WHITE, font_size=32)
        right_title = Text("Predicted", color=WHITE, font_size=32)
        left_title.next_to(left_img, UP, buff=0.3)
        right_title.next_to(right_img, UP, buff=0.3)

        self.play(Write(left_title), Write(right_title))
        self.wait(0.3)
        """

        # --------------------------------------------
        # File lists: replace with your actual files
        # --------------------------------------------
        left_files = [
            "true_density_0.png",
            "true_density_1.png",
            "true_density_2.png",
            "true_density_3.png",
            "true_density_4.png",
        ]

        right_files = [
            "posterior_density_0.png",
            "posterior_density_1.png",
            "posterior_density_2.png",
            "posterior_density_3.png",
            "posterior_density_4.png",
        ]

        # Check files exist
        for f in left_files + right_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Could not find image file: {f}")

        if len(left_files) != len(right_files):
            raise ValueError("Both image lists must have the same number of frames.")


        #self.play(Write(left_title), Write(right_title))
        #self.wait(0.3)

        # --------------------------------------------
        # Helper function
        # --------------------------------------------
        img_width = 4.5   # smaller than before
        img_height = 3.2
        def make_image(filename, center):
            img = ImageMobject(filename)
            img.set_width(img_width)
            img.set_height(img_height)
            img.move_to(center)
            return img



        # --------------------------------------------
        # Initial images
        # --------------------------------------------
        left_img = make_image(left_files[0], left_center)
        right_img = make_image(right_files[0], right_center)

        left_title = Text("True", color=WHITE, font_size=32)
        right_title = Text("Posterior", color=WHITE, font_size=32)
        left_title.next_to(left_img, UP, buff=0.3)
        right_title.next_to(right_img, UP, buff=0.3)

        #self.play(Write(left_title), Write(right_title))
        self.wait(7)

        left_time = Text("t = 0", color=WHITE, font_size=26).next_to(left_title, RIGHT, buff=0.3)
        right_time = Text("t = 0", color=WHITE, font_size=26).next_to(right_title, RIGHT, buff=0.3)

        # Fade in first image only
        self.play(Write(left_title))
        self.wait(0.3)
        self.play(FadeIn(left_img), FadeIn(left_time), run_time=1.0)
        self.wait(0.8)

        # Fade in second image only
        self.play(Write(right_title))
        self.wait(0.3)
        self.play(FadeIn(right_img), FadeIn(right_time), run_time=1.0)
        self.wait(0.8)

        # --------------------------------------------
        # Play both sequences simultaneously
        # --------------------------------------------
        current_left = left_img
        current_right = right_img

        for i in range(1, len(left_files)):
            next_left = make_image(left_files[i], left_center)
            next_right = make_image(right_files[i], right_center)

            next_left_time = Text(f"t = {i}", color=WHITE, font_size=26).move_to(left_time)
            next_right_time = Text(f"t = {i}", color=WHITE, font_size=26).move_to(right_time)

            self.play(
                FadeIn(next_left),
                FadeOut(current_left),
                FadeIn(next_right),
                FadeOut(current_right),
                Transform(left_time, next_left_time),
                Transform(right_time, next_right_time),
                run_time=0.8,
            )

            current_left = next_left
            current_right = next_right
            self.wait(0.25)

        self.wait(1.0)