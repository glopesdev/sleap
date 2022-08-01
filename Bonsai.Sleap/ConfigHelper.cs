﻿using System;
using System.IO;
using System.Linq;
using OpenCV.Net;
using System.Collections.Generic;
using YamlDotNet.RepresentationModel;

namespace Bonsai.Sleap
{
    public static class ConfigHelper
    {
        public static YamlMappingNode OpenFile(string fileName)
        {
            var yaml = new YamlStream();
            var reader = new StringReader(File.ReadAllText(fileName));
            yaml.Load(reader);

            var document = yaml.Documents.FirstOrDefault();
            if (document == null)
            {
                throw new ArgumentException("The specified pose config file is empty.", nameof(fileName));
            }

            var mapping = document.RootNode as YamlMappingNode;
            return mapping;
        }



        public static TrainingConfig LoadTrainingConfig(string fileName)
        {
            var mapping = OpenFile(fileName);
            return LoadTrainingConfig(mapping);
        }

        public static TrainingConfig LoadTrainingConfig(YamlMappingNode mapping)
        {

            var config = new TrainingConfig();
            config.ModelType = GetModelType(mapping);
            ParseModel(config, mapping);

            config.TargetSize = new Size(
                int.Parse((string)mapping["data"]["preprocessing"]["target_width"]),
                int.Parse((string)mapping["data"]["preprocessing"]["target_height"]));
            config.InputScaling = float.Parse((string)mapping["data"]["preprocessing"]["input_scaling"]);
            return config;

        }

        public static ModelType GetModelType(YamlMappingNode mapping)
        {
            int Nfound = 0;
            var availableModels = mapping["model"]["heads"];
            var outArg = ModelType.InvalidModel;

            if (availableModels["single_instance"].AllNodes.Count() > 1)
            {
                Nfound++;
                outArg = ModelType.SingleInstance;
            }
            if (availableModels["centroid"].AllNodes.Count() > 1)
            {
                Nfound++;
                outArg = ModelType.Centroid;
            }
            if (availableModels["centered_instance"].AllNodes.Count() > 1)
            {
                Nfound++;
                outArg = ModelType.CenteredInstance;
            }
            if (availableModels["multi_instance"].AllNodes.Count() > 1)
            {
                Nfound++;
                outArg = ModelType.MultiInstance;
            }

            if (Nfound == 0)
            {
                throw new KeyNotFoundException("No models found in training_config.json file.");
            }
            if (Nfound > 1)
            {
                throw new KeyNotFoundException("Multiple models found in training_config.json file.");
            }
            return outArg;
        }

        public static void ParseModel(TrainingConfig config, YamlMappingNode mapping)
        {
            switch (config.ModelType)
            {
                case ModelType.SingleInstance:
                    Parse_SingleInstance_Model(config, mapping);
                    break;
                case ModelType.Centroid:
                    break;
                case ModelType.CenteredInstance:
                    break;
                case ModelType.MultiInstance:
                    break;
            }
        }

        public static void Parse_SingleInstance_Model(TrainingConfig config, YamlMappingNode mapping)
        {
            var partNames = (YamlSequenceNode)mapping["model"]["heads"]["single_instance"]["part_names"];
            foreach (var part in partNames.Children)
            {
                config.PartNames.Add((string)part);
            }
            AddSkeleton(config, mapping);
        }

        public static void Parse_MultiInstance_Model(TrainingConfig config, YamlMappingNode mapping)
        {
            var partNames = (YamlSequenceNode) mapping["model"]["heads"]["multi_class_topdown"]["confmaps"]["part_names"];
            foreach (var part in partNames.Children)
            {
                config.PartNames.Add((string) part);
            }
            var classNames = (YamlSequenceNode)mapping["model"]["heads"]["multi_class_topdown"]["class_vectors"]["classes"];
            foreach (var id in classNames.Children)
            {
                config.ClassNames.Add((string) id);
            }
            AddSkeleton(config, mapping);
        }

        public static void AddSkeleton(TrainingConfig config, YamlMappingNode mapping)
        {
            var skeleton = new Skeleton();
            skeleton.DirectedEdges = (string)mapping["data"]["labels"]["skeletons"][0]["directed"] == "true";
            skeleton.Name = (string)mapping["data"]["labels"]["skeletons"][0]["graph"]["name"];

            //TODO: fill edges
            var edges = new List<Link>();
            skeleton.Edges = edges;
            config.Skeleton = skeleton;
        }

        public enum ModelType
        {
            InvalidModel = 0,
            SingleInstance = 1,
            Centroid = 2,
            CenteredInstance = 3,
            MultiInstance = 4
        }

    }
}
