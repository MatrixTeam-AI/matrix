# ======================== Stop/Run ============================
python journee/utils/update_ray_var.py --name "current_state" \
                                       --name_space "matrix" --value "RUN"

python journee/utils/update_ray_var.py --name "current_state" \
                                       --name_space "matrix" --value "STOP"

# ======================= Prompt Change(before/during generation) ============================
python journee/utils/update_ray_var.py --name "current_prompt" \
                                       --name_space "matrix" \
                                       --value "a driving car is surrounded by many trees, and the sun is setting"

python journee/utils/update_ray_var.py --name "current_prompt" \
                                       --name_space "matrix" \
                                       --value "a car is driving in a daytime desert with very bright sunshine"

python journee/utils/update_ray_var.py --name "current_prompt" \
                                       --name_space "matrix" \
                                       --value "a car is driving on the road alongside a river with a mountain in the background and a blue sky with white clouds"

python journee/utils/update_ray_var.py --name "current_prompt" \
                                       --name_space "matrix" \
                                       --value "In a barren desert, a white SUV is driving. From an overhead panoramic shot, the vehicle has blue and red stripe decorations on its body, and there is a black spoiler at the rear. It is traversing through sand dunes and shrubs, kicking up a cloud of dust. In the distance, undulating mountains can be seen, with a sky of deep blue and a few white clouds floating by."

python journee/utils/update_ray_var.py --name "current_prompt" \
                                       --name_space "matrix" \
                                       --value "On a lush green meadow, a white car is driving. From an overhead panoramic shot, this car is adorned with blue and red stripes on its body, and it has a black spoiler at the rear. The camera follows the car as it moves through a field of golden wheat, surrounded by green grass and trees. In the distance, a river and some hills can be seen, with a cloudless blue sky above."

python journee/utils/update_ray_var.py --name "current_prompt" \
                                       --name_space "matrix" \
                                       --value "The video shows a white car driving on a country road on a sunny day. The car comes from the back of the scene, moving forward along the road, with open fields and distant hills surrounding it. As the car moves, the vegetation on both sides of the road and distant buildings can be seen. The entire video records the car's journey through the natural environment using a follow-shot technique."