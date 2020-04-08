#plotting code
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
#mAP class array
person_day = []
people_day = []
cyclist_day = []
avg_day = []

person_night = []
people_night = []
cyclist_night = []
avg_night = []


AP_file_night = open("test_data_map_multi_night.txt", 'r')
AP_file_day = open("test_data_map_multi_day.txt", 'r')
contents_night = AP_file_night.readlines()
contents_day = AP_file_day.readlines()

for i, loss in enumerate(contents_day,1):
	data = loss.split(" ")
	person_day.append(float(data[0])*100)
	people_day.append(float(data[1])*100)
	cyclist_day.append(float(data[2])*100)
	avg_day.append(float(data[3])*100)

for i, loss in enumerate(contents_night,1):
	data = loss.split(" ")
	person_night.append(float(data[0])*100)
	people_night.append(float(data[1])*100)
	cyclist_night.append(float(data[2])*100)
	avg_night.append(float(data[3])*100)

# print("person %.2f" % person[i])
fig = plt.figure()
ax = plt.subplot(221)
ax.plot(person_day, marker = "o", linestyle="solid", color = "blue", markersize=0, linewidth=1, label= "day")
ax.plot(person_night, marker = "o", linestyle="dashed", color = "red", markersize=0, linewidth=1, label = "night")
ax.set_title("person")
# ax.axes.get_yaxis().set_visible(False)
ax.axes.set_ylim(top=80, auto=True)
# # ax.tick_params(axis='y',which='major', h=10)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

ax2 = plt.subplot(222)
ax2.plot(people_day, marker = "o", linestyle="solid", color = "blue", markersize=0, linewidth=1, label= "day")
ax2.plot(people_night, marker = "o", linestyle="dashed", color = "red", markersize=0, linewidth=1, label = "night")
ax2.set_title("people")
ax2.axes.set_ylim(top=80, auto=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0.)

ax3 = plt.subplot(223)
ax3.plot(cyclist_day, marker = "o", linestyle="solid", color = "blue", markersize=0, linewidth=1, label= "day")
ax3.plot(cyclist_night, marker = "o", linestyle="dashed", color = "red", markersize=0, linewidth=1, label = "night")
ax3.set_title("cyclist")
# ax3.axes.get_yaxis().set_visible(False)
ax3.axes.set_ylim(top=80, auto=True)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='center', borderaxespad=0.)

ax4 = plt.subplot(224)
ax4.plot(avg_day, marker = "o", linestyle="solid", color = "blue", markersize=0, linewidth=1, label= "day")
ax4.plot(avg_night, marker = "o", linestyle="dashed", color = "red", markersize=0, linewidth=1, label = "night")
ax4.set_title("average")
# ax4.axes.get_yaxis().set_visible(False)
ax4.axes.set_ylim(top=80, auto=True)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='center', borderaxespad=0.)

ax.set_xlabel("Epoch")
ax.set_ylabel("Average Precision (%)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Average Precision (%)")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Average Precision (%)")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Average Precision (%)")

plt.subplots_adjust(hspace = 0.5, wspace = 0.5) #range among plots
fig.suptitle("Precision RGB-Infrared Images")
plt.show()
	